{-# LANGUAGE GADTs               #-}
{-# LANGUAGE PatternGuards       #-}
{-# LANGUAGE QuasiQuotes         #-}
{-# LANGUAGE RankNTypes          #-}
{-# LANGUAGE RecordWildCards     #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TemplateHaskell     #-}
{-# LANGUAGE TypeOperators       #-}
{-# LANGUAGE ViewPatterns        #-}
{-# OPTIONS -fno-warn-name-shadowing #-}
-- |
-- Module      : Data.Array.Accelerate.CUDA.CodeGen
-- Copyright   : [2008..2014] Manuel M T Chakravarty, Gabriele Keller
--               [2009..2014] Trevor L. McDonell
-- License     : BSD3
--
-- Maintainer  : Trevor L. McDonell <tmcdonell@cse.unsw.edu.au>
-- Stability   : experimental
-- Portability : non-portable (GHC extensions)
--

module Data.Array.Accelerate.CUDA.CodeGen (

  CUTranslSkel, codegenAcc, codegenToSeq, codegenFoldSeq, codegenScanSeq,

  codegenAccLifted,

) where

-- new TODO cleanup
import Data.Monoid ( mempty )

-- libraries
import Prelude                                                  hiding ( id, exp, replicate )
import Control.Applicative                                      ( (<$>), (<*>) )
import Control.Monad.State.Strict
import Data.HashSet                                             ( HashSet )
import Foreign.CUDA.Analysis
import Language.C.Quote.CUDA
import qualified Language.C                                     as C
import qualified Data.HashSet                                   as Set

-- friends
import Data.Array.Accelerate.Error
import Data.Array.Accelerate.Type
import Data.Array.Accelerate.Product
import Data.Array.Accelerate.Trafo
import Data.Array.Accelerate.Pretty                             ()
import Data.Array.Accelerate.Analysis.Shape
import Data.Array.Accelerate.Array.Sugar                        ( Array, Shape, Elt, EltRepr
                                                                , Tuple(..), TupleRepr, (:.)
                                                                , Vector, Scalar, Z )
import Data.Array.Accelerate.Array.Representation               ( SliceIndex(..) )
import qualified Data.Array.Accelerate.Array.Sugar              as Sugar
import qualified Data.Array.Accelerate.Analysis.Type            as Sugar

import Data.Array.Accelerate.CUDA.AST                           hiding ( Val(..), prj )
import Data.Array.Accelerate.CUDA.CodeGen.Constant
import Data.Array.Accelerate.CUDA.CodeGen.Base
import Data.Array.Accelerate.CUDA.CodeGen.Type
import Data.Array.Accelerate.CUDA.CodeGen.Monad
import Data.Array.Accelerate.CUDA.CodeGen.Mapping
import Data.Array.Accelerate.CUDA.CodeGen.IndexSpace
import Data.Array.Accelerate.CUDA.CodeGen.PrefixSum
import Data.Array.Accelerate.CUDA.CodeGen.Reduction
import Data.Array.Accelerate.CUDA.CodeGen.Stencil
import Data.Array.Accelerate.CUDA.CodeGen.Streaming
import Data.Array.Accelerate.CUDA.Foreign.Import                ( canExecuteExp )
import qualified Data.Array.Accelerate.CUDA.CodeGen.Arithmetic  as A


-- Local environments
--
data Val env where
  Empty ::                       Val ()
  Push  :: Val env -> [C.Exp] -> Val (env, s)

prj :: Idx env t -> Val env -> [C.Exp]
prj ZeroIdx      (Push _   v) = v
prj (SuccIdx ix) (Push val _) = prj ix val
prj _            _            = $internalError "prj" "inconsistent valuation"


-- Array expressions
-- -----------------

-- | Instantiate an array computation with a set of concrete function and type
-- definitions to fix the parameters of an algorithmic skeleton. The generated
-- code can then be pretty-printed to file, and compiled to object code
-- executable on the device. This generates a set of __global__ device functions
-- required to compute the given computation node.
--
-- The code generator requires that the only array form allowed within scalar
-- expressions are array variables. The list of array-valued scalar inputs are
-- taken as the environment.
--
-- TODO: include a measure of how much shared memory a kernel requires.
--

seqShape :: (Shape sh, Elt e) => Gamma senv -> Idx senv (Array sh e) -> [C.Exp]
seqShape senv idx = [cexp| chunksz |] : cshape (reifyDim idx) (fst (namesOfSvar senv idx))

codegenSeq :: forall sh e senv aenv. (Shape sh, Elt e) => DeviceProperties -> Gamma aenv -> Gamma senv -> Idx senv (Array sh e) -> CUDA (CUDelayedAcc senv aenv (sh :. Int) e)
codegenSeq _ _ senv (idx :: Idx senv (Array sh e)) = return $
  CUDelayed (CUExp ([], sh))
            (CUFun1 elim inst)
            (CUFun1 elim instL)
  where
    (shname, arrs) = namesOfSvar senv idx

    sh :: [C.Exp]
    sh = seqShape senv idx
    elim   = zip (repeat True)
    get ix = ([], map (\a -> [cexp| $id:a [ $exp:ix ] |]) arrs)
    inst  :: Instantiate1 (sh :. Int) e
    inst  ix  = get (ctoIndex sh ix) -- TODO let-bind to avoid code-duplication.
    instL :: Instantiate1 Int e
    instL [i] = get (rvalue i)
    instL _ = $internalError "codegenSeq" $ "instL"

codegenDelayedAcc :: forall aenv sh e. (Shape sh, Elt e) => DeviceProperties -> Gamma aenv -> DelayedOpenAcc aenv (Array sh e) -> CUDA (CUDelayedAcc () aenv sh e)
codegenDelayedAcc dev aenv = travD
  where
    travD :: DelayedOpenAcc aenv (Array sh e) -> CUDA (CUDelayedAcc () aenv sh e)
    travD Manifest{}  = $internalError "codegenAcc" "expected delayed array"
    travD Delayed{..} = CUDelayed <$> travE extentD
                                  <*> travF1 indexD
                                  <*> travF1 linearIndexD

    travF1 :: DelayedFun () aenv (a -> b) -> CUDA (CUFun1 () aenv (a -> b))
    travF1 = codegenFun1 dev aenv mempty (error "#1")

    travE :: DelayedExp () aenv t -> CUDA (CUExp () aenv t)
    travE = codegenExp dev aenv mempty (error "#2")

-- Given a function from shape 'sh' to some type, provide a funtion
-- from the shape of a on chunk 'sh :. Int' that simply ignores the
-- streaming dimension.
liftedShapeFun :: CUFun1 senv aenv (sh'          -> e)
               -> CUFun1 senv aenv ((sh' :. Int) -> e)
liftedShapeFun (CUFun1 used fun) =
  CUFun1 (\ xs -> (False, head xs) : used (tail xs))
         (\ xs -> fun (tail xs))

-- Given a permutation (sh -> sh'), provide a permutation on chunks
-- (sh :. Int -> sh' :. Int) that is the identity in the streaming
-- dimension.
liftedPermute :: CUFun1 senv aenv (sh'        -> sh       )
                 -> CUFun1 senv aenv (sh' :. Int -> sh :. Int)
liftedPermute (CUFun1 used fun) =
  CUFun1 (\ xs -> (True, head xs) : used (tail xs))
         (\ xs -> let (blocks, exps) = fun (tail xs) in (blocks, rvalue (head xs) : exps))

replicateCUFun :: (Shape sh, Shape sl) => SliceIndex slix (EltRepr sl) co (EltRepr sh)
               -> CUFun1 senv aenv (sh -> sl)
replicateCUFun slix =
  CUFun1 (zip (reverse (used slix)))
         (\ xs -> ([], reverse $ restrict slix (reverse (map rvalue xs))))
  where
    used :: SliceIndex a b c d -> [Bool]
    used SliceNil = []
    used (SliceFixed sl) = False : used sl
    used (SliceAll   sl) = True  : used sl

    restrict :: SliceIndex a b c d -> [e]{-sh-} -> [e]{-sl-}
    restrict SliceNil        []     = []
    restrict (SliceFixed sl) (_:ys) =     restrict sl ys
    restrict (SliceAll   sl) (y:ys) = y : restrict sl ys
    restrict _ _ = error "replicateCUFun"

sliceCUFun :: (Shape sh, Shape sl, Elt slix) => SliceIndex (EltRepr slix) (EltRepr sl) co (EltRepr sh)
           -> CUExp () aenv slix -> CUFun1 senv aenv (sl -> sh)
sliceCUFun slix (CUExp (blocks, exps)) =
  CUFun1 (zip (repeat True))
         (\ xs -> (blocks, reverse $ combine slix (reverse (map rvalue xs)) (reverse exps)))
  where
    combine :: SliceIndex a b c d -> [e]{-slix-} -> [e]{-sl-} -> [e]{-sh-}
    combine SliceNil [] [] = []
    combine (SliceFixed sl) xs (y:ys) = y : combine sl xs ys
    combine (SliceAll   sl) (x:xs) ys = x : combine sl xs ys
    combine _ _ _ = error "sliceCUFun"

flattenCU :: CUDelayedAcc senv aenv (sh :. Int :. Int) e
          -> CUDelayedAcc senv aenv (sh :. Int) e
flattenCU (CUDelayed (CUExp (shB, sh)) _ c)
  = CUDelayed  (CUExp (shB, sh' ++ [[cexp| $exp:n * $exp:k |]] )) ($internalError "flattenCU" "linear indexing only") c
  where
    n = head sh
    k = last sh
    sh' = init (tail sh)

codegenAccLifted :: forall senv aenv sh e.
                    DeviceProperties
                 -> PreOpenArrayOp (Idx senv) DelayedOpenAcc senv aenv (Array sh e)
                 -> Gamma aenv
                 -> Gamma senv
                 -> [ CUTranslSkel senv aenv (Chunk sh e) ]
codegenAccLifted dev op aenv senv
  = codegen
  $ case op of
      -- Producers
      Map f x                   -> mkMap dev aenv senv <$> travF1streaming (reifyDim x) f <*> travD x

      Generate sh f             -> mkGenerate dev aenv senv  <$> (liftedShapeFun <$> travF1streaming (expDim sh) f)
      Transform sh p f x        -> mkTransform dev aenv senv <$> (liftedPermute <$> travF1streaming (expDim sh) p)
                                                             <*> travF1streaming (expDim sh) f
                                                             <*> travD x
      Backpermute sh p x        -> mkTransform dev aenv senv <$> (liftedPermute <$> travF1streaming (expDim sh) p) <*> travF1streaming (expDim sh) id <*> travD x
      -- Consumers
      Fold f z a                -> mkFold  dev aenv senv  <$> travF2 f <*> travE z  <*> travD a
      Fold1 f a                 -> mkFold1 dev aenv senv  <$> travF2 f <*> travD a
      FoldSeg f z a s           -> map cast <$> (mkFoldSeg dev aenv senv <$> travF2 f <*> travE z  <*> (flattenCU <$> travD a) <*> (flattenCU <$> travD s))
      Fold1Seg f a s            -> map cast <$> (mkFold1Seg dev aenv senv <$> travF2 f <*> (flattenCU <$> travD a)  <*> (flattenCU <$> travD s))
--      Scanl f z a               -> mkScanl dev aenv     <$> travF2 f <*> travE z  <*> travD a
--      Scanr f z a               -> mkScanr dev aenv     <$> travF2 f <*> travE z  <*> travD a
--      Scanl' f z a              -> mkScanl' dev aenv    <$> travF2 f <*> travE z  <*> travD a
--      Scanr' f z a              -> mkScanr' dev aenv    <$> travF2 f <*> travE z  <*> travD a
--      Scanl1 f a                -> mkScanl1 dev aenv    <$> travF2 f <*> travD a
--      Scanr1 f a                -> mkScanr1 dev aenv    <$> travF2 f <*> travD a
      Permute f d p a           -> mkPermute dev aenv senv  <$> travF2 f <*> (liftedPermute <$> travF1streaming (reifyDim d) p) <*> travD a
--      Stencil f b a             -> mkStencil dev aenv   <$> travF1 f <*> travB a b
--      Stencil2 f b1 a1 b2 a2    -> mkStencil2 dev aenv  <$> travF2 f <*> travB a1 b1 <*> travB a2 b2
      Replicate slix _ a        -> mkTransform dev aenv senv (liftedPermute (replicateCUFun slix)) <$> travF1streaming 0 id <*> travD a
      Slice slix a slice        -> mkTransform dev aenv senv <$> (liftedPermute <$> (sliceCUFun slix <$> travE slice)) <*> travF1streaming 0 id <*> travD a
      ZipWith{}                 -> fusionError

  where
    eltSz :: Name -> Int -> C.Exp
    eltSz name n = csize (cshape n name)

    cast :: CUTranslSkel senv aenv a -> CUTranslSkel senv aenv b
    cast (CUTranslSkel entry code) = CUTranslSkel entry code

    id :: Elt a => DelayedFun senv aenv (a -> a)
    id = Lam (Body (Var ZeroIdx))

    travF1streaming :: Int -> DelayedFun senv aenv (a -> b) -> CUDA (CUFun1 senv aenv (a -> b))
    travF1streaming n = codegenFun1 dev aenv senv (eltSz "Out" n)

    travD :: (Shape sh', Elt e') => Idx senv (Array sh' e') -> CUDA (CUDelayedAcc senv aenv (sh' :. Int) e')
    travD = codegenSeq dev aenv senv
{-
    travF1 :: DelayedFun () aenv (a -> b) -> CUDA (CUFun1 () aenv (a -> b))
    travF1 = codegenFun1 dev aenv mempty undefined
-}
    travF2 :: DelayedFun () aenv (a -> b -> c) -> CUDA (CUFun2 () aenv (a -> b -> c))
    travF2 = codegenFun2 dev aenv mempty (error "#3")

    travE :: DelayedExp () aenv t -> CUDA (CUExp () aenv t)
    travE = codegenExp dev aenv mempty (error "#4")
{-
    travB :: forall sh e. Elt e
          => Idx senv (Array sh e) -> Boundary (EltRepr e) -> CUDA (Boundary (CUExp () aenv e))
    travB _ Clamp        = return Clamp
    travB _ Mirror       = return Mirror
    travB _ Wrap         = return Wrap
    travB _ (Constant c) = return . Constant $ CUExp ([], constant (Sugar.eltType (undefined::e)) c)
-}
    -- caffeine and misery
    prim :: String
    prim                = showPreArrOp op
    fusionError         = $internalError "codegenAcc" $ "unexpected fusible material: " ++ prim

codegen :: CUDA [CUTranslSkel senv aenv a] -> [CUTranslSkel senv aenv a]
codegen cuda =
  let (skeletons, st)                = runCUDA cuda
      addTo (CUTranslSkel name code) =
        CUTranslSkel name (Set.foldr (\h c -> [cedecl| $esc:("#include \"" ++ h ++ "\"") |] : c) code (headers st))
  in
  map addTo skeletons

codegenAcc :: forall aenv arrs.
              DeviceProperties
           -> PreOpenArrayOp (DelayedOpenAcc aenv) DelayedOpenAcc () aenv arrs
           -> Gamma aenv
           -> [ CUTranslSkel () aenv arrs ]
codegenAcc dev op aenv
  = codegen
  $ case op of
      -- Producers
      Map f a                   -> mkMap dev aenv senv       <$> travF1 f <*> travD a
      Generate _ f              -> mkGenerate dev aenv senv  <$> travF1 f
      Transform _ p f a         -> mkTransform dev aenv senv <$> travF1 p <*> travF1 f  <*> travD a
      Backpermute _ p a         -> mkTransform dev aenv senv <$> travF1 p <*> travF1 id <*> travD a

      -- Consumers
      Fold f z a                -> mkFold  dev aenv senv     <$> travF2 f <*> travE z  <*> travD a
      Fold1 f a                 -> mkFold1 dev aenv senv     <$> travF2 f <*> travD a
      FoldSeg f z a s           -> mkFoldSeg dev aenv senv   <$> travF2 f <*> travE z  <*> travD a <*> travD s
      Fold1Seg f a s            -> mkFold1Seg dev aenv senv  <$> travF2 f <*> travD a  <*> travD s
      Scanl f z a               -> mkScanl dev aenv     <$> travF2 f <*> travE z  <*> travD a
      Scanr f z a               -> mkScanr dev aenv     <$> travF2 f <*> travE z  <*> travD a
      Scanl' f z a              -> mkScanl' dev aenv    <$> travF2 f <*> travE z  <*> travD a
      Scanr' f z a              -> mkScanr' dev aenv    <$> travF2 f <*> travE z  <*> travD a
      Scanl1 f a                -> mkScanl1 dev aenv    <$> travF2 f <*> travD a
      Scanr1 f a                -> mkScanr1 dev aenv    <$> travF2 f <*> travD a
      Permute f _ p a           -> mkPermute dev aenv senv  <$> travF2 f <*> travF1 p <*> travD a
      Stencil f b a             -> mkStencil dev aenv   <$> travF1 f <*> travB a b
      Stencil2 f b1 a1 b2 a2    -> mkStencil2 dev aenv  <$> travF2 f <*> travB a1 b1 <*> travB a2 b2
      Replicate{}               -> fusionError
      Slice{}                   -> fusionError
      ZipWith{}                 -> fusionError

  where
    senv :: Gamma ()
    senv = mempty

    id :: Elt a => DelayedFun () aenv (a -> a)
    id = Lam (Body (Var ZeroIdx))

    -- code generation for delayed arrays
    travD :: (Shape sh, Elt e) => DelayedOpenAcc aenv (Array sh e) -> CUDA (CUDelayedAcc () aenv sh e)
    travD = codegenDelayedAcc dev aenv

    -- scalar code generation
    travF1 :: DelayedFun () aenv (a -> b) -> CUDA (CUFun1 () aenv (a -> b))
    travF1 = codegenFun1 dev aenv senv (error "#5")

    travF2 :: DelayedFun () aenv (a -> b -> c) -> CUDA (CUFun2 () aenv (a -> b -> c))
    travF2 = codegenFun2 dev aenv senv (error "#6")

    travE :: DelayedExp () aenv t -> CUDA (CUExp () aenv t)
    travE = codegenExp dev aenv senv (error "#7")

    travB :: forall sh e. Elt e
          => DelayedOpenAcc aenv (Array sh e) -> Boundary (EltRepr e) -> CUDA (Boundary (CUExp () aenv e))
    travB _ Clamp        = return Clamp
    travB _ Mirror       = return Mirror
    travB _ Wrap         = return Wrap
    travB _ (Constant c) = return . Constant $ CUExp ([], constant (Sugar.eltType (undefined::e)) c)

    -- caffeine and misery
    prim :: String
    prim                = showPreArrOp op
    fusionError         = $internalError "codegenAcc" $ "unexpected fusible material: " ++ prim


codegenFoldSeq :: forall aenv e. Elt e
               => DeviceProperties
               -> Gamma aenv
               -> DelayedFun () aenv (e -> e -> e)
               -> CUTranslSkel () aenv (Chunk Z e)
codegenFoldSeq dev aenv f =
    head
  $ codegen
  $ return
  <$> (mkInplaceUpdate dev aenv <$> codegenFun2 dev aenv mempty (error "#9") f)

codegenScanSeq :: forall senv aenv e. Elt e
               => DeviceProperties
               -> Gamma aenv
               -> DelayedFun () aenv (e -> e -> e)
               -> Idx senv (Scalar e)
               -> [CUTranslSkel senv aenv (Chunk Z e)]
codegenScanSeq dev aenv f x =
    codegen
  $ mkScanl dev aenv <$> codegenFun2 dev aenv mempty (error "#10") f <*> return z <*> codegenSeq dev aenv mempty x
  where
    (_, _, sum) = readArray "Sum" (undefined :: Vector e)
    z :: CUExp () aenv e
    z | CUDelayed _ _ (CUFun1 _ get) <- sum 
      = let (_, sh, _) = locals "sh" (undefined :: Z :. Int) 
        in CUExp (get sh)

codegenToSeq :: forall aenv senv slix sl co sh e. (Shape sl, Shape sh, Elt e)
                => SliceIndex slix
                              (EltRepr sl)
                              co
                              (EltRepr sh)
                -> DeviceProperties
                -> DelayedOpenAcc aenv (Array sh e)
                -> Gamma aenv
                -> CUTranslSkel senv aenv (Chunk sl e)
codegenToSeq slix dev acc aenv = 
  head $ codegen $ return <$> (mkToSeq slix dev aenv <$> codegenDelayedAcc dev aenv acc)
{-
    travE :: forall t. DelayedExp () aenv t -> CUDA (CUExp () aenv t)
    travE = codegenExp dev aenv mempty undefined

    travF1 :: forall a b. DelayedFun () aenv (a -> b) -> CUDA (CUFun1 () aenv (a -> b))
    travF1 = codegenFun1 dev aenv mempty undefined
-}

-- Scalar function abstraction
-- ---------------------------

-- Generate code for scalar function abstractions.
--
-- This is quite awkward: we have an outer monad to generate fresh variable
-- names, but since we know that even if the function in applied many times (for
-- example, collective operations such as 'fold' and 'scan'), the variables will
-- not shadow each other. Thus, we don't need fresh names at _every_ invocation
-- site, so we hack this a bit to return a pure closure.
--
-- Note that the implementation of def-use analysis used for dead code
-- elimination requires that we always generate code for closed functions.
-- Additionally, we require two passes over the function: once when performing
-- the analysis, and a second time when instantiating the function in the
-- skeleton.
--
codegenFun1
    :: forall aenv senv a b. DeviceProperties
    -> Gamma aenv
    -> Gamma senv
    -> C.Exp -- Stream index
    -> DelayedFun senv aenv (a -> b)
    -> CUDA (CUFun1 senv aenv (a -> b))
codegenFun1 dev aenv senv k fun
  | Lam (Body f) <- fun
  = let
        go :: Rvalue x => [x] -> Gen ([C.BlockItem], [C.Exp])
        go x = do
          code  <- mapM use =<< codegenOpenExp dev aenv senv k f (Empty `Push` map rvalue x)
          env'  <- getEnv
          return (env', code)

        -- Initial code generation proceeds with dummy variable names. The real
        -- names are substituted later when we instantiate the skeleton.
        (_,u,_) = locals "undefined_x" (undefined :: a)
    in do
      n                 <- get
      ExpST _ used      <- execCGM (go u)
      return $ CUFun1 (mark used u)
             $ \xs -> evalState (evalCGM (go xs)) n
  --
  | otherwise
  = $internalError "codegenFun1" "expected unary function"


codegenFun2
    :: forall aenv senv a b c. DeviceProperties
    -> Gamma aenv
    -> Gamma senv
    -> C.Exp -- Stream index
    -> DelayedFun senv aenv (a -> b -> c)
    -> CUDA (CUFun2 senv aenv (a -> b -> c))
codegenFun2 dev aenv senv k fun
  | Lam (Lam (Body f)) <- fun
  = let
        go :: (Rvalue x, Rvalue y) => [x] -> [y] -> Gen ([C.BlockItem], [C.Exp])
        go x y = do
          code  <- mapM use =<< codegenOpenExp dev aenv senv k f (Empty `Push` map rvalue x `Push` map rvalue y)
          env'  <- getEnv
          return (env', code)

        (_,u,_)  = locals "undefined_x" (undefined :: a)
        (_,v,_)  = locals "undefined_y" (undefined :: b)
    in do
      n                 <- get
      ExpST _ used      <- execCGM (go u v)
      return $ CUFun2 (mark used u) (mark used v)
             $ \xs ys -> evalState (evalCGM (go xs ys)) n
  --
  | otherwise
  = $internalError "codegenFun2" "expected binary function"


-- It is important to filter output terms of a function that will not be used.
-- Consider this pattern from the map kernel:
--
--   items:(x      .=. get ix)
--   items:(set ix .=. f x)
--
-- If this is applied to the following expression where we extract the first
-- component of a 4-tuple:
--
--   map (\t -> let (x,_,_,_) = unlift t in x) vec4
--
-- Then the first line 'get ix' still reads all four components of the input
-- vector, even though only one is used. Conversely, if we directly apply the
-- data fetch to f, then the redundant reads are eliminated, but this is simply
-- inlining the read into the function body, so if the argument is used multiple
-- times so to is the data read multiple times.
--
-- The procedure for determining which variables are used is to record each
-- singleton expression produced throughout code generation to a set. It doesn't
-- matter if the expression is a variable (which we are interested in) or
-- something else. Once generation completes, we can test which of the input
-- variables also appear in the output set. Later, we integrate this information
-- when assigning to l-values: if the variable is not in the set, simply elide
-- that statement.
--
-- In the above map example, this means that the usage data is taken from 'f',
-- but applies to which results of 'get ix' are committed to memory.
--
mark :: HashSet C.Exp -> [C.Exp] -> ([a] -> [(Bool,a)])
mark used xs
  = let flags = map (\x -> x `Set.member` used) xs
    in  zipWith (,) flags

visit :: [C.Exp] -> Gen [C.Exp]
visit exp
  | [x] <- exp  = use x >> return exp
  | otherwise   =          return exp


-- Scalar expressions
-- ------------------

-- Generation of scalar expressions
--
codegenExp :: DeviceProperties -> Gamma aenv -> Gamma senv -> C.Exp -> DelayedExp senv aenv t -> CUDA (CUExp senv aenv t)
codegenExp dev aenv senv k exp =
  evalCGM $ do
    code        <- codegenOpenExp dev aenv senv k exp Empty
    env         <- getEnv
    return      $! CUExp (env,code)


-- The core of the code generator, buildings lists of untyped C expression
-- fragments. This is tricky to get right!
--
codegenOpenExp
    :: forall aenv senv env' t'. DeviceProperties
    -> Gamma aenv
    -> Gamma senv
    -> C.Exp -- Stream index
    -> DelayedOpenExp env' senv aenv t'
    -> Val env'
    -> Gen [C.Exp]
codegenOpenExp dev aenv senv k = cvtE
  where
    -- Generate code for a scalar expression in depth-first order. We run under
    -- a monad that generates fresh names and keeps track of let bindings.
    --
    cvtE :: forall env t. DelayedOpenExp env senv aenv t -> Val env -> Gen [C.Exp]
    cvtE exp env = visit =<<
      case exp of
        Let bnd body            -> elet bnd body env
        Var ix                  -> return $ prj ix env
        PrimConst c             -> return $ [primConst c]
        Const c                 -> return $ constant (Sugar.eltType (undefined::t)) c
        PrimApp f x             -> primApp f x env
        Tuple t                 -> cvtT t env
        Prj i t                 -> prjT i t exp env
        Cond p t e              -> cond p t e env
        While p f x             -> while p f x env

        -- Shapes and indices
        IndexNil                -> return []
        IndexAny                -> return []
        IndexCons sh sz         -> (++) <$> cvtE sh env <*> cvtE sz env
        IndexHead ix            -> return . cindexHead <$> cvtE ix env
        IndexTail ix            ->          cindexTail <$> cvtE ix env
        IndexSlice ix slix sh   -> indexSlice ix slix sh env
        IndexFull  ix slix sl   -> indexFull  ix slix sl env
        ToIndex sh ix           -> toIndex   sh ix env
        FromIndex sh ix         -> fromIndex sh ix env

        -- Arrays and indexing
        Index acc ix            -> index acc ix env
        LinearIndex acc ix      -> linearIndex acc ix env
        Shape acc               -> shape acc env
        ShapeSize sh            -> shapeSize sh env
        Intersect sh1 sh2       -> intersect sh1 sh2 env
        Union sh1 sh2           -> union sh1 sh2 env

        -- Sequence operations
        IndexS x ix             -> indexS x ix env
        LinearIndexS x ix       -> linearIndexS x ix env
        ShapeS x                -> shapeS x env

        --Foreign function
        Foreign ff _ e          -> foreignE ff e env

    -- The heavy lifting
    -- -----------------

    -- Scalar let expressions evaluate their terms and generate new (const)
    -- variable bindings to store these results. These are carried the monad
    -- state, which also gives us a supply of fresh names. The new names are
    -- added to the environment for use in the body via the standard Var term.
    --
    -- Note that we have not restricted the scope of these new bindings: once
    -- something is added, it remains in scope forever. We are relying on
    -- liveness analysis of the CUDA compiler to manage register pressure.
    --
    elet :: DelayedOpenExp env senv aenv bnd -> DelayedOpenExp (env, bnd) senv aenv body -> Val env -> Gen [C.Exp]
    elet bnd body env = do
      bnd'      <- cvtE bnd env >>= pushEnv bnd
      body'     <- cvtE body (env `Push` bnd')
      return body'

    -- When evaluating primitive functions, we evaluate each argument to the
    -- operation as a statement expression. This is necessary to ensure proper
    -- short-circuit behaviour for logical operations.
    --
    primApp :: PrimFun (a -> b) -> DelayedOpenExp env senv aenv a -> Val env -> Gen [C.Exp]
    primApp f x env =
      case f of
        -- operators from Num
        PrimAdd{}               -> binary A.add x env
        PrimSub{}               -> binary A.sub x env
        PrimMul{}               -> binary A.mul x env
        PrimNeg{}               -> unary A.negate x env
        PrimAbs ty              -> unary (A.abs ty) x env
        PrimSig ty              -> unaryM (A.signum ty) x env
        -- operators from Integral & Bits
        PrimQuot{}              -> binary A.quot x env
        PrimRem{}               -> binary A.rem x env
        PrimQuotRem ty          -> binaryM2 (A.quotRem ty) x env
        PrimIDiv ty             -> binaryM (A.idiv ty) x env
        PrimMod ty              -> binaryM (A.mod ty) x env
        PrimDivMod ty           -> binaryM2 (A.divMod ty) x env
        PrimBAnd{}              -> binary A.band x env
        PrimBOr{}               -> binary A.bor x env
        PrimBXor{}              -> binary A.xor x env
        PrimBNot{}              -> unary A.bnot x env
        PrimBShiftL{}           -> binary A.shiftL x env
        PrimBShiftR{}           -> binary A.shiftRA x env
        PrimBRotateL ty         -> binaryM (A.rotateL ty) x env
        PrimBRotateR ty         -> binaryM (A.rotateR ty) x env
        -- operators from Fractional and Floating
        PrimFDiv{}              -> binary A.fdiv x env
        PrimRecip ty            -> unary (A.recip ty) x env
        PrimSin ty              -> unary (A.sin ty) x env
        PrimCos ty              -> unary (A.cos ty) x env
        PrimTan ty              -> unary (A.tan ty) x env
        PrimAsin ty             -> unary (A.asin ty) x env
        PrimAcos ty             -> unary (A.acos ty) x env
        PrimAtan ty             -> unary (A.atan ty) x env
        PrimAsinh ty            -> unary (A.asinh ty) x env
        PrimAcosh ty            -> unary (A.acosh ty) x env
        PrimAtanh ty            -> unary (A.atanh ty) x env
        PrimExpFloating ty      -> unary (A.exp ty) x env
        PrimSqrt ty             -> unary (A.sqrt ty) x env
        PrimLog ty              -> unary (A.log ty) x env
        PrimFPow ty             -> binary (A.pow ty) x env
        PrimLogBase ty          -> binary (A.logBase ty) x env
        -- operators from RealFrac
        PrimTruncate ta tb      -> unary (A.trunc ta tb) x env
        PrimRound ta tb         -> unary (A.round ta tb) x env
        PrimFloor ta tb         -> unary (A.floor ta tb) x env
        PrimCeiling ta tb       -> unary (A.ceiling ta tb) x env
        -- operators from RealFloat
        PrimAtan2 ty            -> binary (A.atan2 ty) x env
        PrimIsNaN{}             -> unary A.isNaN x env
        -- relational and equality operators
        PrimLt{}                -> binary A.lt x env
        PrimGt{}                -> binary A.gt x env
        PrimLtEq{}              -> binary A.leq x env
        PrimGtEq{}              -> binary A.geq x env
        PrimEq{}                -> binary A.eq x env
        PrimNEq{}               -> binary A.neq x env
        PrimMax ty              -> binary (A.max ty) x env
        PrimMin ty              -> binary (A.min ty) x env
        -- logical operators
        PrimLAnd                -> binary A.land x env
        PrimLOr                 -> binary A.lor x env
        PrimLNot                -> unary A.lnot x env
        -- type conversions
        PrimOrd                 -> unary A.ord x env
        PrimChr                 -> unary A.chr x env
        PrimBoolToInt           -> unary A.boolToInt x env
        PrimFromIntegral ta tb  -> unary (A.fromIntegral ta tb) x env
      where
        cvtE' :: DelayedOpenExp env senv aenv a -> Val env -> Gen C.Exp
        cvtE' e env = do
          (b,r) <- clean $ single "primApp" <$> cvtE e env
          if null b
             then return r
             else return [cexp| ({ $items:b; $exp:r; }) |]

        -- TLM: This is a bit ugly. Consider making all primitive functions from
        --      Arithmetic.hs evaluate in the Gen monad.
        --
        unary :: (C.Exp -> C.Exp) -> DelayedOpenExp env senv aenv a -> Val env -> Gen [C.Exp]
        unary f = unaryM  (return . f)

        unaryM :: (C.Exp -> Gen C.Exp) -> DelayedOpenExp env senv aenv a -> Val env -> Gen [C.Exp]
        unaryM f a env = do
          a' <- cvtE' a env
          r  <- f a'
          return [r]

        binary :: (C.Exp -> C.Exp -> C.Exp) -> DelayedOpenExp env senv aenv (a,b) -> Val env -> Gen [C.Exp]
        binary f = binaryM (\a b -> return (f a b))

        binaryM :: (C.Exp -> C.Exp -> Gen C.Exp) -> DelayedOpenExp env senv aenv (a,b) -> Val env -> Gen [C.Exp]
        binaryM f (Tuple (NilTup `SnocTup` a `SnocTup` b)) env = do
          a' <- cvtE' a env
          b' <- cvtE' b env
          r  <- f a' b'
          return [r]
        binaryM _ _ _ = $internalError "primApp" "unexpected argument to binary function"

        binaryM2 :: (C.Exp -> C.Exp -> Gen (C.Exp, C.Exp)) -> DelayedOpenExp env senv aenv (a,b) -> Val env -> Gen [C.Exp]
        binaryM2 f (Tuple (NilTup `SnocTup` a `SnocTup` b)) env = do
          a'    <- cvtE' a env
          b'    <- cvtE' b env
          (r,s) <- f a' b'
          return [r,s]
        binaryM2 _ _ _ = $internalError "primApp" "unexpected argument to binary function"

    -- Convert an open expression into a sequence of C expressions. We retain
    -- snoc-list ordering, so the element at tuple index zero is at the end of
    -- the list. Note that nested tuple structures are flattened.
    --
    cvtT :: Tuple (DelayedOpenExp env senv aenv) t -> Val env -> Gen [C.Exp]
    cvtT tup env =
      case tup of
        NilTup          -> return []
        SnocTup t e     -> (++) <$> cvtT t env <*> cvtE e env

    -- Project out a tuple index. Since the nested tuple structure is flattened,
    -- this actually corresponds to slicing out a subset of the list of C
    -- expressions, rather than picking out a single element.
    --
    prjT :: forall env t e. TupleIdx (TupleRepr t) e
         -> DelayedOpenExp env senv aenv t
         -> DelayedOpenExp env senv aenv e
         -> Val env
         -> Gen [C.Exp]
    prjT ix t e env =
      let subset = reverse
                 . take (length      $ expType e)
                 . drop (prjToInt ix $ Sugar.preExpType Sugar.delayedAccType t)
                 . reverse
      in
      subset <$> cvtE t env

    -- Convert a tuple index into the corresponding integer. Since the internal
    -- representation is flat, be sure to walk over all sub components when indexing
    -- past nested tuples.
    --
    prjToInt :: TupleIdx t e -> TupleType a -> Int
    prjToInt ZeroTupIdx     _                 = 0
    prjToInt (SuccTupIdx i) (b `PairTuple` a) = sizeTupleType a + prjToInt i b
    prjToInt _              _                 = $internalError "prjToInt" "inconsistent valuation"

    sizeTupleType :: TupleType a -> Int
    sizeTupleType UnitTuple       = 0
    sizeTupleType (SingleTuple _) = 1
    sizeTupleType (PairTuple a b) = sizeTupleType a + sizeTupleType b

    -- Scalar conditionals insert a standard if/else statement block. We don't
    -- use the ternary expression operator (?:) because this forces all
    -- auxiliary bindings for both the true and false branches to always be
    -- evaluated before the correct result is chosen.
    --
    cond :: forall env t. Elt t
         => DelayedOpenExp env senv aenv Bool
         -> DelayedOpenExp env senv aenv t
         -> DelayedOpenExp env senv aenv t
         -> Val env
         -> Gen [C.Exp]
    cond p t f env = do
      p'        <- cvtE p env
      ok        <- single "Cond" <$> pushEnv p p'
      ifTrue    <- clean $ cvtE t env
      ifFalse   <- clean $ cvtE f env

      -- Generate names for the result variables, which will be initialised
      -- within each branch of the conditional. Twiddle the names a bit to
      -- avoid clobbering.
      var_r     <- lift fresh
      let (_, r, declr) = locals ('l':var_r) (undefined :: t)
          branch        = [citem| if ( $exp:ok ) {
                                      $items:(r .=. ifTrue)
                                  }
                                  else {
                                      $items:(r .=. ifFalse)
                                  } |]
                        : map C.BlockDecl declr

      modify (\s -> s { localBindings = branch ++ localBindings s })
      return r

    -- Value recursion
    --
    while :: forall env a. Elt a
          => DelayedOpenFun env senv aenv (a -> Bool)        -- continue while predicate returns true
          -> DelayedOpenFun env senv aenv (a -> a)           -- loop body
          -> DelayedOpenExp env senv aenv a                  -- initial value
          -> Val env
          -> Gen [C.Exp]
    while test step x env
      | Lam (Body p)    <- test
      , Lam (Body f)    <- step
      = do
           -- Generate code for the initial value, then bind this to a fresh
           -- (mutable) variable. We need build the declarations ourselves, and
           -- twiddle the names a bit to avoid clobbering.
           --
           x'           <- cvtE x env
           var_acc      <- lift fresh
           var_ok       <- lift fresh
           var_tmp      <- lift fresh

           let (_, acc, decl_acc) = locals ('l':var_acc) (undefined :: a)
               (_, ok,  decl_ok)  = locals ('l':var_ok)  (undefined :: Bool)
               (tmp, _, _)        = locals ('l':var_tmp) (undefined :: a)

           -- Generate code for the predicate and body expressions, with the new
           -- names baked in directly. We can't use 'codegenFun1', because
           -- def-use analysis won't be able to see into this new function.
           --
           -- However, we do need to generate the function with a clean set of
           -- local bindings, and extract and new declarations afterwards.
           --
           p'   <- clean $ cvtE p (env `Push` acc)
           f'   <- clean $ cvtE f (env `Push` acc)

           -- Piece it all together. Note that declarations are added to the
           -- localBindings in reverse order. Also, we have to be careful not to
           -- assign the results of f' direction into acc. Why? If some of the
           -- variables in acc are referenced in f', then we risk overwriting
           -- values that are still needed to computer f'.
           --
           let loop = [citem| while ( $exp:(single "while" ok) ) {
                                  $items:(tmp .=. f')
                                  $items:(acc .=. tmp)
                                  $items:(ok  .=. p')
                              } |]
                    : reverse (ok  .=. p')
                   ++ reverse (acc .=. x')
                   ++ map C.BlockDecl decl_ok
                   ++ map C.BlockDecl decl_acc

           modify (\s -> s { localBindings = loop ++ localBindings s })
           return acc

      | otherwise
      = error "Would you say we'd be venturing into a zone of danger?"

    -- Restrict indices based on a slice specification. In the SliceAll case we
    -- elide the presence of IndexAny from the head of slx, as this is not
    -- represented in by any C term (Any ~ [])
    --
    indexSlice :: SliceIndex (EltRepr slix) sl co (EltRepr sh)
               -> DelayedOpenExp env senv aenv slix
               -> DelayedOpenExp env senv aenv sh
               -> Val env
               -> Gen [C.Exp]
    indexSlice sliceIndex slix sh env =
      let restrict :: SliceIndex slix sl co sh -> [C.Exp] -> [C.Exp] -> [C.Exp]
          restrict SliceNil              _       _       = []
          restrict (SliceAll   sliceIdx) slx     (sz:sl) = sz : restrict sliceIdx slx sl
          restrict (SliceFixed sliceIdx) (_:slx) ( _:sl) =      restrict sliceIdx slx sl
          restrict _ _ _ = $internalError "IndexSlice" "unexpected shapes"
          --
          slice slix' sh' = reverse $ restrict sliceIndex (reverse slix') (reverse sh')
      in
      slice <$> cvtE slix env <*> cvtE sh env

    -- Extend indices based on a slice specification. In the SliceAll case we
    -- elide the presence of Any from the head of slx.
    --
    indexFull :: SliceIndex (EltRepr slix) (EltRepr sl) co sh
              -> DelayedOpenExp env senv aenv slix
              -> DelayedOpenExp env senv aenv sl
              -> Val env
              -> Gen [C.Exp]
    indexFull sliceIndex slix sl env =
      let extend :: SliceIndex slix sl co sh -> [C.Exp] -> [C.Exp] -> [C.Exp]
          extend SliceNil              _        _       = []
          extend (SliceAll   sliceIdx) slx      (sz:sh) = sz : extend sliceIdx slx sh
          extend (SliceFixed sliceIdx) (sz:slx) sh      = sz : extend sliceIdx slx sh
          extend _ _ _ = $internalError "IndexFull" "unexpected shapes"
          --
          replicate slix' sl' = reverse $ extend sliceIndex (reverse slix') (reverse sl')
      in
      replicate <$> cvtE slix env <*> cvtE sl env

    -- Convert between linear and multidimensional indices
    --
    toIndex :: DelayedOpenExp env senv aenv sh -> DelayedOpenExp env senv aenv sh -> Val env -> Gen [C.Exp]
    toIndex sh ix env = do
      sh'   <- mapM use =<< cvtE sh env
      ix'   <- mapM use =<< cvtE ix env
      return [ ctoIndex sh' ix' ]

    fromIndex :: DelayedOpenExp env senv aenv sh -> DelayedOpenExp env senv aenv Int -> Val env -> Gen [C.Exp]
    fromIndex sh ix env = do
      sh'   <- mapM use =<< cvtE sh env
      ix'   <- cvtE ix env
      tmp   <- lift fresh
      let (ls, sz) = cfromIndex sh' (single "fromIndex" ix') tmp
      modify (\st -> st { localBindings = reverse ls ++ localBindings st })
      return sz

    -- Project out a single scalar element from an array. The array expression
    -- does not contain any free scalar variables (strictly flat data
    -- parallelism) and has been floated out to be replaced by an array index.
    --
    -- As we have a non-parametric array representation, be sure to bind the
    -- linear array index as it will be used to access each component of a
    -- tuple.
    --
    -- Note that after evaluating the linear array index we bind this to a fresh
    -- variable of type 'int', so there is an implicit conversion from
    -- Int -> Int32.
    --
    index :: (Shape sh, Elt e)
          => DelayedOpenAcc aenv (Array sh e)
          -> DelayedOpenExp env senv aenv sh
          -> Val env
          -> Gen [C.Exp]
    index acc ix env
      | Manifest (Avar idx) <- acc
      = let (sh, arr)   = namesOfAvar aenv idx
            ty          = accType acc
        in do
        ix'     <- mapM use =<< cvtE ix env
        i       <- bind cint $ ctoIndex (cshape (expDim ix) sh) ix'
        return   $ zipWith (\t a -> indexArray dev t (cvar a) i) ty arr
      --
      | otherwise
      = $internalError "Index" "expected array variable"


    linearIndex :: (Shape sh, Elt e)
                => DelayedOpenAcc aenv (Array sh e)
                -> DelayedOpenExp env senv aenv Int
                -> Val env
                -> Gen [C.Exp]
    linearIndex acc ix env
      | Manifest (Avar idx) <- acc
      = let (_, arr)    = namesOfAvar aenv idx
            ty          = accType acc
        in do
        ix'     <- mapM use =<< cvtE ix env
        i       <- bind [cty| int |] $ single "LinearIndex" ix'
        return   $ zipWith (\t a -> indexArray dev t (cvar a) i) ty arr
      --
      | otherwise
      = $internalError "LinearIndex" "expected array variable"

    -- Array shapes created in this method refer to the shape of free array
    -- variables. As such, they are always passed as arguments to the kernel,
    -- not computed as part of the scalar expression. These shapes are
    -- transferred to the kernel as a structure, and so the individual fields
    -- need to be "unpacked", to work with our handling of tuple structures.
    --
    shape :: (Shape sh, Elt e) => DelayedOpenAcc aenv (Array sh e) -> Val env -> Gen [C.Exp]
    shape acc _env
      | Manifest (Avar idx) <- acc
      = return $ cshape (delayedDim acc) (fst (namesOfAvar aenv idx))

      | otherwise
      = $internalError "Shape" "expected array variable"

    -- The size of a shape, as the product of the extent in each dimension. The
    -- definition is inlined, but we could also call the C function helpers.
    --
    shapeSize :: DelayedOpenExp env senv aenv sh -> Val env -> Gen [C.Exp]
    shapeSize sh env = return . csize <$> cvtE sh env

    -- Intersection of two shapes, taken as the minimum in each dimension.
    --
    intersect :: forall env sh. Elt sh
              => DelayedOpenExp env senv aenv sh
              -> DelayedOpenExp env senv aenv sh
              -> Val env -> Gen [C.Exp]
    intersect sh1 sh2 env =
      zipWith (\a b -> ccall "min" [a,b]) <$> cvtE sh1 env <*> cvtE sh2 env

    -- Union of two shapes, taken as the maximum in each dimension.
    --
    union :: forall env sh. Elt sh
          => DelayedOpenExp env senv aenv sh
          -> DelayedOpenExp env senv aenv sh
          -> Val env -> Gen [C.Exp]
    union sh1 sh2 env =
      zipWith (\a b -> ccall "max" [a,b]) <$> cvtE sh1 env <*> cvtE sh2 env


    -- Project out a single scalar element from an array. The array expression
    -- does not contain any free scalar variables (strictly flat data
    -- parallelism) and has been floated out to be replaced by an array index.
    --
    -- As we have a non-parametric array representation, be sure to bind the
    -- linear array index as it will be used to access each component of a
    -- tuple.
    --
    -- Note that after evaluating the linear array index we bind this to a fresh
    -- variable of type 'int', so there is an implicit conversion from
    -- Int -> Int32.
    --
    indexS :: forall sh e env. (Shape sh, Elt e)
           => Idx senv (Array sh e)
           -> DelayedOpenExp env senv aenv sh
           -> Val env
           -> Gen [C.Exp]
    indexS idx ix env
      = let (_, arr)   = namesOfSvar senv idx
            ty         = eltType (undefined :: e)
        in do
        ix'     <- mapM use =<< (cchunkIndex k :) <$> cvtE ix env
        i       <- bind cint $ ctoIndex (seqShape senv idx) ix'
        return   $ zipWith (\t a -> indexArray dev t (cvar a) i) ty arr

    linearIndexS :: forall sh e env. (Shape sh, Elt e)
                 => Idx senv (Array sh e)
                 -> DelayedOpenExp env senv aenv Int
                 -> Val env
                 -> Gen [C.Exp]
    linearIndexS idx ix env
      = let (_, arr)    = namesOfSvar senv idx
            ty          = eltType (undefined :: e)
        in do
        ix'     <- mapM use =<< cvtE ix env
        i       <- bind [cty| int |] $ single "LinearIndex" ix'
        i'      <- bind cint $ [cexp| $exp:i + ($exp:k * ($exp:(cchunkIndex k))) |]
        return   $ zipWith (\t a -> indexArray dev t (cvar a) i') ty arr

    -- Array shapes created in this method refer to the shape of free array
    -- variables. As such, they are always passed as arguments to the kernel,
    -- not computed as part of the scalar expression. These shapes are
    -- transferred to the kernel as a structure, and so the individual fields
    -- need to be "unpacked", to work with our handling of tuple structures.
    --
    shapeS :: (Shape sh, Elt e) => Idx senv (Array sh e) -> Val env -> Gen [C.Exp]
    shapeS idx _env
      = return (tail (seqShape senv idx))

    -- Foreign scalar functions. We need to extract any header files that might
    -- be required so they can be added to the top level definitions.
    --
    -- Additionally, we insert an explicit type cast from the foreign function
    -- result back into Accelerate types (c.f. Int vs int).
    --
    foreignE :: forall f a b env. (Sugar.Foreign f, Elt a, Elt b)
             => f a b
             -> DelayedOpenExp env senv aenv a
             -> Val env
             -> Gen [C.Exp]
    foreignE ff x env = case canExecuteExp ff of
      Nothing      -> $internalError "codegenOpenExp" "Non-CUDA foreign expression encountered"
      Just (hs, f) -> do
        lift $ modify (\st -> st { headers = foldl (flip Set.insert) (headers st) hs })
        args    <- cvtE x env
        mapM_ use args
        return  $  [ccall f (ccastTup (Sugar.eltType (undefined::a)) args)]

    -- Execute a command in a new environment. The old environment is replaced
    -- on exit, and the result and any new bindings generated are returned.
    --
    clean :: Gen a -> Gen ([C.BlockItem], a)
    clean this = do
      env  <- state (\s -> ( localBindings s, s { localBindings = []  } ))
      r    <- this
      env' <- state (\s -> ( localBindings s, s { localBindings = env } ))
      return (reverse env', r)

    -- Some terms demand we extract only singly typed expressions
    --
    single :: String -> [C.Exp] -> C.Exp
    single _   [x] = x
    single loc _   = $internalError loc "expected single expression"


-- Auxiliary Functions
-- -------------------

ccast :: ScalarType a -> C.Exp -> C.Exp
ccast ty x = [cexp| ($ty:(typeOf ty)) $exp:x |]

ccastTup :: TupleType e -> [C.Exp] -> [C.Exp]
ccastTup ty = fst . travTup ty
  where
    travTup :: TupleType e -> [C.Exp] -> ([C.Exp], [C.Exp])
    travTup UnitTuple         xs     = ([], xs)
    travTup (SingleTuple ty') (x:xs) = ([ccast ty' x], xs)
    travTup (PairTuple l r)   xs     =
      let (ls, xs' ) = travTup l xs
          (rs, xs'') = travTup r xs'
      in (ls ++ rs, xs'')
    travTup _ _ = $internalError "ccastTup" "not enough expressions to match type"
