{-# LANGUAGE BangPatterns               #-}
{-# LANGUAGE CPP                        #-}
{-# LANGUAGE FlexibleContexts           #-}
{-# LANGUAGE FlexibleInstances          #-}
{-# LANGUAGE GADTs                      #-}
{-# LANGUAGE IncoherentInstances        #-}
{-# LANGUAGE NoForeignFunctionInterface #-}
{-# LANGUAGE PatternGuards              #-}
{-# LANGUAGE ScopedTypeVariables        #-}
{-# LANGUAGE TemplateHaskell            #-}
{-# LANGUAGE TupleSections              #-}
{-# LANGUAGE TypeOperators              #-}
{-# LANGUAGE TypeSynonymInstances       #-}
{-# LANGUAGE UndecidableInstances       #-}
{-# LANGUAGE ScopedTypeVariables        #-}
-- |
-- Module      : Data.Array.Accelerate.CUDA.Execute
-- Copyright   : [2008..2014] Manuel M T Chakravarty, Gabriele Keller
--               [2009..2014] Trevor L. McDonell
-- License     : BSD3
--
-- Maintainer  : Trevor L. McDonell <tmcdonell@cse.unsw.edu.au>
-- Stability   : experimental
-- Portability : non-portable (GHC extensions)
--

module Data.Array.Accelerate.CUDA.Execute (

  -- * Execute a computation under a CUDA environment
  executeAcc, executeAfun1,

  -- * Executing a sequence computation and streaming its output.
  LazySeq(..), streamSeq,

) where

-- TODO cleanup
import Data.Typeable
import Data.Monoid ( mempty )

-- friends
import Data.Array.Accelerate.CUDA.AST
import Data.Array.Accelerate.CUDA.State
import Data.Array.Accelerate.CUDA.Array.Data
import Data.Array.Accelerate.CUDA.Array.Slice                   ( copyArgs )
import Data.Array.Accelerate.CUDA.Array.Sugar
import Data.Array.Accelerate.CUDA.Foreign.Import                ( canExecuteAcc )
import Data.Array.Accelerate.CUDA.CodeGen.Base                  ( Name, namesOfArray, groupOfInt )
import Data.Array.Accelerate.CUDA.Execute.Event                 ( Event )
import Data.Array.Accelerate.CUDA.Execute.Stream                ( Stream )
import qualified Data.Array.Accelerate.CUDA.Array.Prim          as Prim
import qualified Data.Array.Accelerate.CUDA.Debug               as D
import qualified Data.Array.Accelerate.CUDA.Execute.Event       as Event
import qualified Data.Array.Accelerate.CUDA.Execute.Stream      as Stream

import Data.Array.Accelerate.Error
import Data.Array.Accelerate.Interpreter                        ( evalPrim, evalPrimConst, evalPrj )
import Data.Array.Accelerate.Array.Data                         ( ArrayElt, ArrayData )
import Data.Array.Accelerate.Array.Representation               ( SliceIndex(..) )
import Data.Array.Accelerate.FullList                           ( FullList(..), List(..) )
import Data.Array.Accelerate.Lifetime                           ( withLifetime )
import Data.Array.Accelerate.Trafo                              ( Extend(..) )
import qualified Data.Array.Accelerate.Array.Representation     as R


-- standard library
import Prelude                                                  hiding ( exp, sum, iterate )
import Control.Applicative                                      hiding ( Const )
import Control.Monad                                            ( join, when, liftM )
import Control.Monad.Reader                                     ( asks )
import Control.Monad.State                                      ( gets )
import Control.Monad.Trans                                      ( MonadIO, liftIO )
import Control.Monad.Trans.Cont                                 ( ContT(..), evalContT )
import System.IO.Unsafe                                         ( unsafeInterleaveIO, unsafePerformIO )
import Data.Int
import Data.Word

import Foreign.CUDA.Analysis.Device                             ( computeCapability, Compute(..) )
import qualified Foreign.CUDA.Driver                            as CUDA
import qualified Data.HashMap.Strict                            as Map


-- Asynchronous kernel execution
-- -----------------------------

-- A suspended sequence computation.
data LazySeq a = Done | Yield [a] (CIO (LazySeq a))

indexLast :: Shape sh => (sh :. Int) -> Int
indexLast = last . shapeToList

indexInit :: Shape sh => (sh :. Int) -> sh
indexInit = listToShape . init . shapeToList

infixr 3 .:
(.:) :: Shape sh => Int -> sh -> (sh :. Int)
(.:) n sh = listToShape (shapeToList sh ++ [n])

type Val' senv = (Sval senv, Int)

-- Valuation for an environment of sequence windows.
--
data Sval senv where
  Sempty :: Sval ()
  Spush  :: Sval senv -> Async (PrimChunk sh e) -> Sval (senv, Array sh e)

-- Projection of a window from a window valuation using a de Bruijn
-- index.
--
sprj :: Idx senv (Array sh e) -> Sval senv -> Async (PrimChunk sh e)
sprj ZeroIdx       (Spush _   v) = v
sprj (SuccIdx idx) (Spush val _) = sprj idx val
sprj _             _             = $internalError "prj" "inconsistent valuation"



sval :: Val' senv -> Sval senv
sval = fst

senvSize :: Val' senv -> Int
senvSize = snd


-- Projection of a window from a window valuation using a de Bruijn
-- index.
--
prj' :: Idx senv (Array sh e) -> Val' senv -> Async (PrimChunk sh e)
prj' x (senv, _) = sprj x senv

push' :: Val' senv -> Async (PrimChunk sh e) -> Val' (senv, Array sh e)
push' (senv, n) c = (Spush senv c, n)

-- Arrays with an associated CUDA Event that will be signalled once the
-- computation has completed.
--
data Async a = Async {-# UNPACK #-} !Event !a

instance Functor Async where
  fmap f (Async ev a) = Async ev (f a)

-- Valuation for an environment of asynchronous array computations
--
data Aval env where
  Aempty :: Aval ()
  Apush  :: Aval env -> Async t -> Aval (env, t)

-- Projection of a value from a valuation using a de Bruijn index.
--
aprj :: Idx env t -> Aval env -> Async t
aprj ZeroIdx       (Apush _   x) = x
aprj (SuccIdx idx) (Apush val _) = aprj idx val
aprj _             _             = $internalError "aprj" "inconsistent valuation"

-- All work submitted to the given stream will occur after the asynchronous
-- event for the given array has been fulfilled. Synchronisation is performed
-- efficiently on the device. This function returns immediately.
--
after :: MonadIO m => Stream -> Async a -> m a
after stream (Async event arr) = liftIO $ Event.after event stream >> return arr

-- Block the calling thread until the event for the given array computation
-- is recorded.
--
wait :: MonadIO m => Async a -> m a
wait (Async e x) = liftIO $ Event.block e >> return x


-- Execute the given computation in a unique execution stream.
--
streaming :: (Stream -> CIO a) -> (Async a -> CIO b) -> CIO b
streaming first second = do
  context   <- asks activeContext
  reservoir <- gets streamReservoir
  table     <- gets eventTable
  Stream.streaming context reservoir table first (\e a -> second (Async e a))


-- Execute the given computation in a unique execution stream.
-- TODO
streaming' :: (Stream -> CIO a) -> (Async a -> CIO () -> CIO b) -> CIO b
streaming' first second = do
  context   <- asks activeContext
  reservoir <- gets streamReservoir
  table     <- gets eventTable
  Stream.streaming' context reservoir table first (\e a -> second (Async e a))


-- Array expression evaluation
-- ---------------------------

-- Computations are evaluated by traversing the AST bottom-up, and for each node
-- distinguishing between three cases:
--
-- 1. If it is a Use node, return a reference to the device memory holding the
--    array data
--
-- 2. If it is a non-skeleton node, such as a let-binding or shape conversion,
--    this is executed directly by updating the environment or similar
--
-- 3. If it is a skeleton node, the associated binary object is retrieved,
--    memory allocated for the result, and the kernel(s) that implement the
--    skeleton are invoked
--

executeAcc :: Arrays a => ExecAcc a -> CIO a
executeAcc !acc = streaming (executeOpenAcc acc Aempty) wait

executeAfun1 :: (Arrays a, Arrays b) => ExecAfun (a -> b) -> a -> CIO b
executeAfun1 !afun !arrs = do
  streaming (useArrays (arrays arrs) (fromArr arrs))
            (\(Async event ()) -> executeOpenAfun1 afun Aempty (Async event arrs))
  where
    useArrays :: ArraysR arrs -> arrs -> Stream -> CIO ()
    useArrays ArraysRunit         ()       _  = return ()
    useArrays (ArraysRpair r1 r0) (a1, a0) st = useArrays r1 a1 st >> useArrays r0 a0 st
    useArrays ArraysRarray        arr      st = useArrayAsync arr (Just st)


executeOpenAfun1 :: PreOpenAfun ExecOpenAcc aenv (a -> b) -> Aval aenv -> Async a -> CIO b
executeOpenAfun1 (Alam (Abody f)) aenv x = streaming (executeOpenAcc f (aenv `Apush` x)) wait
executeOpenAfun1 _                _    _ = error "the sword comes out after you swallow it, right?"

executeOpenAfun2 :: PreOpenAfun ExecOpenAcc aenv (a -> b -> c) -> Aval aenv -> Async a -> Async b -> CIO c
executeOpenAfun2 (Alam (Alam (Abody f))) aenv x y = streaming (executeOpenAcc f (aenv `Apush` x `Apush` y)) wait
executeOpenAfun2 _                       _    _ _ = error "the sword comes out after you swallow it, right?"


-- Evaluate an open array computation
--
executeOpenAcc
    :: forall aenv arrs.
       ExecOpenAcc aenv arrs
    -> Aval aenv
    -> Stream
    -> CIO arrs
executeOpenAcc EmbedAcc{} _ _
  = $internalError "execute" "unexpected delayed array"
executeOpenAcc (ExecSeq l)                                !aenv !stream
  = executeSequence defaultSeqConfig l aenv stream
executeOpenAcc (ExecAcc (FL () kernel more) !gamma !pacc) !aenv !stream
  = case pacc of

      -- Array introduction
      Use arr                   -> return (toArr arr)
      Unit x                    -> newArray Z . const =<< travE x
      Reshape sh a              -> reshapeOp <$> travE sh <*> travA a

      -- Environment manipulation
      Avar ix                   -> after stream (aprj ix aenv)
      Alet bnd body             -> streaming (executeOpenAcc bnd aenv) (\x -> executeOpenAcc body (aenv `Apush` x) stream)
      Apply f a                 -> streaming (executeOpenAcc a aenv)   (executeOpenAfun1 f aenv)
      Atuple tup                -> toAtuple <$> travT tup
      Aprj ix tup               -> evalPrj ix . fromAtuple <$> travA tup
      Acond p t e               -> travE p >>= \x -> if x then travA t else travA e
      Awhile p f a              -> awhile p f =<< travA a

      -- Foreign
      Aforeign ff afun a        -> aforeign ff afun =<< travA a

      Collect{}                 -> streamingError
      ArrayOp op                ->
        case op of

          -- Producers
          Map _ a                   -> executeOp =<< extent a
          Generate sh _             -> executeOp =<< travE sh
          Transform sh _ _ _        -> executeOp =<< travE sh
          Backpermute sh _ _        -> executeOp =<< travE sh

          -- Consumers
          Fold _ _ a                -> foldOp  =<< extent a
          Fold1 _ a                 -> fold1Op =<< extent a
          FoldSeg _ _ a s           -> join $ foldSegOp <$> extent a <*> extent s
          Fold1Seg _ a s            -> join $ foldSegOp <$> extent a <*> extent s
          Scanl1 _ a                -> scan1Op =<< extent a
          Scanr1 _ a                -> scan1Op =<< extent a
          Scanl' _ _ a              -> scan'Op =<< extent a
          Scanr' _ _ a              -> scan'Op =<< extent a
          Scanl _ _ a               -> scanOp True  =<< extent a
          Scanr _ _ a               -> scanOp False =<< extent a
          Permute _ d _ a           -> join $ permuteOp <$> extent a <*> travA d
          Stencil _ _ a             -> stencilOp =<< travA a
          Stencil2 _ _ a1 _ a2      -> join $ stencil2Op <$> travA a1 <*> travA a2

          -- AST nodes that should be inaccessible at this point
          Replicate{}               -> fusionError
          Slice{}                   -> fusionError
          ZipWith{}                 -> fusionError

  where
    fusionError    = $internalError "executeOpenAcc" "unexpected fusible matter"
    streamingError = $internalError "executeOpenAcc" "unexpected sequence computation"

    -- term traversals
    travA :: ExecOpenAcc aenv a -> CIO a
    travA !acc = executeOpenAcc acc aenv stream

    travE :: ExecExp () aenv t -> CIO t
    travE !exp = executeExp exp aenv stream

    travT :: Atuple (ExecOpenAcc aenv) t -> CIO t
    travT NilAtup          = return ()
    travT (SnocAtup !t !a) = (,) <$> travT t <*> travA a

    awhile :: PreOpenAfun ExecOpenAcc aenv (a -> Scalar Bool) -> PreOpenAfun ExecOpenAcc aenv (a -> a) -> a -> CIO a
    awhile p f a = do
      nop <- join $ liftIO <$> (Event.create <$> asks activeContext <*> gets eventTable)
       -- ^ record event never call, so this is a functional no-op
      r   <- executeOpenAfun1 p aenv (Async nop a)
      ok  <- indexArray r 0                     -- TLM TODO: memory manager should remember what is already on the host
      if ok then awhile p f =<< executeOpenAfun1 f aenv (Async nop a)
            else return a

    aforeign :: (Arrays as, Arrays bs, Foreign f) => f as bs -> PreAfun ExecOpenAcc (as -> bs) -> as -> CIO bs
    aforeign ff pureFun a =
      case canExecuteAcc ff of
        Just cudaFun -> cudaFun stream a
        Nothing      -> executeAfun1 pureFun a

    -- get the extent of an embedded array
    extent :: Shape sh => ExecOpenAcc aenv (Array sh e) -> CIO sh
    extent ExecAcc{}     = $internalError "executeOpenAcc" "expected delayed array"
    extent ExecSeq{}     = $internalError "executeOpenAcc" "expected delayed array"
    extent (EmbedAcc sh) = travE sh

    -- Skeleton implementation
    -- -----------------------

    -- Execute a skeleton that has no special requirements: thread decomposition
    -- is based on the given shape.
    --
    executeOp :: (Shape sh, Elt e) => sh -> CIO (Array sh e)
    executeOp !sh = do
      out       <- allocateArray sh
      execute kernel gamma aenv mempty (Sempty, 1) (size sh) out stream
      return out

    -- Change the shape of an array without altering its contents. This does not
    -- execute any kernel programs.
    --
    reshapeOp :: Shape sh => sh -> Array sh' e -> Array sh e
    reshapeOp sh (Array sh' adata)
      = $boundsCheck "reshape" "shape mismatch" (size sh == R.size sh')
      $ Array (fromElt sh) adata

    -- Executing fold operations depend on whether we are recursively collapsing
    -- to a single value using multiple thread blocks, or a multidimensional
    -- single-pass reduction where there is one block per inner dimension.
    --
    fold1Op :: (Shape sh, Elt e) => (sh :. Int) -> CIO (Array sh e)
    fold1Op !sh@(_ :. sz)
      = $boundsCheck "fold1" "empty array" (sz > 0)
      $ foldCore sh

    foldOp :: (Shape sh, Elt e) => (sh :. Int) -> CIO (Array sh e)
    foldOp !(!sh :. sz)
      = foldCore ((listToShape . map (max 1) . shapeToList $ sh) :. sz)

    foldCore :: forall sh e. (Shape sh, Elt e) => (sh :. Int) -> CIO (Array sh e)
    foldCore !(!sh :. sz)
      | dim sh > 0              = executeOp sh
      | Just (Refl :: sh :~: Z) <- eqT -- FMMA TODO change back to "otherwise" (see version on Github)
      = let !numElements        = sz
            (_,!numBlocks,_)    = configure kernel numElements
        in do
          out   <- allocateArray (Z :. numBlocks)
          execute kernel gamma aenv mempty (Sempty, 1) numElements out stream
          foldRec out

    -- Recursive step(s) of a multi-block reduction
    --
    foldRec :: Elt e => Vector e -> CIO (Scalar e)
    foldRec arr@(Array _ !adata)
      | Cons _ rec _ <- more
      = let Z :. sz             = shape arr
            !numElements        = sz
            (_,!numBlocks,_)    = configure rec numElements
        in if sz <= 1
              then return $ Array () adata
              else do
                out     <- allocateArray (Z :. numBlocks)
                execute rec gamma aenv mempty (Sempty, 1) numElements (out, arr) stream
                foldRec out

      | otherwise
      = $internalError "foldRec" "missing phase-2 kernel module"

    -- Segmented reduction. Subtract one from the size of the segments vector as
    -- this is the result of an exclusive scan to calculate segment offsets.
    --
    foldSegOp :: (Shape sh, Elt e) => (sh :. Int) -> (Z :. Int) -> CIO (Array (sh :. Int) e)
    foldSegOp (!sh :. _) !(Z :. sz) = executeOp (sh :. sz - 1)

    scanCore
        :: forall e. Elt e
        => Int
        -> Vector e                     -- to fix Elt vs. EltRepr
        -> Prim.DevicePtrs (EltRepr e)
        -> Prim.DevicePtrs (EltRepr e)
        -> CIO ()
    scanCore = scanCore' (FL () kernel more) gamma aenv stream


    -- Scans, all variations on a theme.
    --
    scanOp :: Elt e => Bool -> (Z :. Int) -> CIO (Vector e)
    scanOp !left !(Z :. numElements) = do
      arr@(Array _ adata)       <- allocateArray (Z :. numElements + 1)
      withDevicePtrs adata (Just stream) $ \out -> do
        let (!body, !sum)
              | left      = (out, advancePtrsOfArrayData adata numElements out)
              | otherwise = (advancePtrsOfArrayData adata 1 out, out)
        --
        scanCore numElements arr body sum
        return arr

    scan1Op :: forall e. Elt e => (Z :. Int) -> CIO (Vector e)
    scan1Op !(Z :. numElements) = do
      arr@(Array _ adata)       <- allocateArray (Z :. numElements + 1) :: CIO (Vector e)
      withDevicePtrs adata (Just stream) $ \body -> do
        let sum {- to fix type -} =  advancePtrsOfArrayData adata numElements body
        --
        scanCore numElements arr body sum
        return (Array ((),numElements) adata)

    scan'Op :: forall e. Elt e => (Z :. Int) -> CIO (Vector e, Scalar e)
    scan'Op !(Z :. numElements) = do
      vec@(Array _ ad_vec)      <- allocateArray (Z :. numElements) :: CIO (Vector e)
      sum@(Array _ ad_sum)      <- allocateArray Z                  :: CIO (Scalar e)
      withDevicePtrs ad_vec (Just stream) $ \d_vec ->
        withDevicePtrs ad_sum (Just stream) $ \d_sum -> do
          --
          scanCore numElements vec d_vec d_sum
          return (vec, sum)

    -- Forward permutation
    --
    permuteOp :: forall sh sh' e. (Shape sh, Shape sh', Elt e) => sh -> Array sh' e -> CIO (Array sh' e)
    permuteOp !sh !dfs = do
      let sh'   = shape dfs
          n'    = size sh'

      out               <- allocateArray sh'
      Array _ locks     <- allocateArray sh'            :: CIO (Array sh' Int32)
      withDevicePtrs locks (Just stream) $ \d_locks -> do

        liftIO $ CUDA.memsetAsync d_locks n' 0 (Just stream)      -- TLM: overlap these two operations?
        copyArrayAsync dfs out (Just stream)
        execute kernel gamma aenv mempty (Sempty, 1) (size sh) (out, d_locks) stream
        return out

    -- Stencil operations. NOTE: the arguments to 'namesOfArray' must be the
    -- same as those given in the function 'mkStencil[2]'.
    --
    stencilOp :: forall sh a b. (Shape sh, Elt a, Elt b) => Array sh a -> CIO (Array sh b)
    stencilOp !arr = do
      let sh    =  shape arr
      out       <- allocateArray sh
      dev       <- asks deviceProperties

      if computeCapability dev < Compute 2 0
         then marshalAccTex (namesOfArray "Stencil" (undefined :: a)) kernel arr (Just stream) $
                execute kernel gamma aenv mempty (Sempty, 1) (size sh) (out, sh) stream
         else execute kernel gamma aenv mempty (Sempty, 1) (size sh) (out, arr) stream
      execute kernel gamma aenv mempty (Sempty, 1) (size sh) (out, arr) stream
      --
      return out

    stencil2Op :: forall sh a b c. (Shape sh, Elt a, Elt b, Elt c)
               => Array sh a -> Array sh b -> CIO (Array sh c)
    stencil2Op !arr1 !arr2
      | Cons _ spec _ <- more
      = let sh1         =  shape arr1
            sh2         =  shape arr2
            (sh, op)
              | fromElt sh1 == fromElt sh2      = (sh1,                 spec)
              | otherwise                       = (sh1 `intersect` sh2, kernel)
        in do
          out   <- allocateArray sh
          dev   <- asks deviceProperties

          if computeCapability dev < Compute 2 0
             then marshalAccTex (namesOfArray "Stencil1" (undefined :: a)) op arr1 (Just stream) $
                  marshalAccTex (namesOfArray "Stencil2" (undefined :: b)) op arr2 (Just stream) $
                  execute op gamma aenv mempty (Sempty, 1) (size sh) (out, sh1,  sh2) stream
             else execute op gamma aenv mempty (Sempty, 1) (size sh) (out, arr1, arr2) stream
          execute op gamma aenv mempty (Sempty, 1) (size sh) (out, arr1, arr2) stream
          --
          return out

      | otherwise
      = $internalError "stencil2Op" "missing stencil specialisation kernel"


scanCore'
        :: forall aenv e a. Elt e
        => FullList () (AccKernel a)   -- executable binary objects
        -> Gamma aenv                  -- free array variables the kernel needs access to
        -> Aval aenv
        -> Stream
        -> Int
        -> Vector e                     -- to fix Elt vs. EltRepr
        -> Prim.DevicePtrs (EltRepr e)
        -> Prim.DevicePtrs (EltRepr e)
        -> CIO ()
scanCore' (FL () kernel more) !gamma !aenv !stream !numElements (Array _ !adata) !body !sum
      | Cons _ !upsweep1 (Cons _ !upsweep2 _) <- more
      = let (_,!numIntervals,_) = configure kernel numElements
            !d_body             = marshalDevicePtrs adata body
            !d_sum              = marshalDevicePtrs adata sum
        in do
          blk   <- allocateArray (Z :. numIntervals) :: CIO (Vector e)

          -- Phase 1: Split the array over multiple thread blocks and calculate
          --          the final scan result from each interval.
          --
          when (numIntervals > 1) $ do
            execute upsweep1 gamma aenv mempty (Sempty, 1) numElements blk stream
            execute upsweep2 gamma aenv mempty (Sempty, 1) numIntervals (blk, blk, d_sum) stream

          -- Phase 2: Re-scan the input using the carry-in value from each
          --          interval sum calculated in phase 1.
          --
          execute kernel gamma aenv mempty (Sempty, 1) numElements (numElements, d_body, blk, d_sum) stream

      | otherwise
      = $internalError "scanOp" "missing multi-block kernel module(s)"



-- Configuration for sequence evaluation.
--
data SeqConfig = SeqConfig
  { chunkSize :: Int -- Allocation limit for a sequence in
                     -- words. Actual runtime allocation should be the
                     -- maximum of this size and the size of the
                     -- largest element in the sequence.
  }

-- Default sequence evaluation configuration for testing purposes.
--
defaultSeqConfig :: SeqConfig
defaultSeqConfig = SeqConfig { chunkSize = case unsafePerformIO (D.queryFlag D.chunk_size) of Nothing -> 128; Just n -> n }


-- An executable stream DAG for executing sequence expressions in a
-- streaming fashion.
--
data StreamDAG senv arrs where
  StreamProducer :: (Shape sh, Elt e) => PrimChunk sh e -> StreamProducer senv (Array sh e) -> StreamDAG (senv, Array sh e) arrs -> StreamDAG senv  arrs
  StreamConsumer :: Arrays a          =>                   StreamConsumer senv a                                                 -> StreamDAG senv  a
  StreamReify    :: (Shape sh, Elt e) =>                   Idx senv (Array sh e)                                                 -> StreamDAG senv  [Array sh e]

instance Show (StreamDAG senv arrs) where
  show s =
    case s of
      StreamProducer c p d -> ".. := " ++
        (case p of
           StreamStreamIn arrs -> "streamIn " ++ show (length arrs)
           StreamMap _ -> "streamMap"
           StreamMapFin n _ -> "streamMapB " ++ show n
        ) ++ ";\n" ++ show d
      StreamConsumer c -> show c ++ ";\n"

instance Show (StreamConsumer senv a) where
  show c = 
        case c of
           StreamFold _ _ _ -> "streamFold"
           StreamStuple NilAtup -> "()"
           StreamStuple (SnocAtup NilAtup c) -> "(" ++ show c ++ ")"
           StreamStuple (SnocAtup (SnocAtup NilAtup c1) c0) -> "(" ++ show c0 ++ "," ++ show c1 ++ ")"
           StreamStuple (SnocAtup (SnocAtup (SnocAtup NilAtup c2) c1) c0) -> "(" ++ show c0 ++ "," ++ show c1 ++ "," ++ show c2 ++ ")"
           StreamStuple t -> "streamStuple"
           
  
-- An executable producer.
--
data StreamProducer senv a where
  StreamStreamIn :: [Array sh e]
                 -> StreamProducer senv (Array sh e)

  StreamMap :: (Val' senv -> Int -> Stream -> CIO ())
            -> StreamProducer senv (Array sh e)

  StreamMapFin :: Int 
             -> (Val' senv -> Int -> Stream -> CIO ())
             -> StreamProducer senv (Array sh e)

-- An executable consumer.
--
data StreamConsumer senv a where

  -- Stream reduction skeleton.
  StreamFold :: (Val' senv -> s -> Stream -> CIO s)   -- Chunk consumer function.
             -> (s -> Stream -> CIO r)                -- Finalizer function.
             -> s                                     -- Accumulator (internal state).
             -> StreamConsumer senv r

  StreamStuple :: IsAtuple a
               => Atuple (StreamConsumer senv) (TupleRepr a)
               -> StreamConsumer senv a

newtype Id a = Id a

data Shapes senv where
  EmptyShapes :: Shapes ()
  PushShape   :: Shapes senv -> sh -> Shapes (senv, Array sh e)

prjShape :: Idx senv (Array sh e) -> Shapes senv -> sh
prjShape ZeroIdx       (PushShape _   v) = v
prjShape (SuccIdx idx) (PushShape val _) = prjShape idx val
prjShape _             _             = $internalError "prj" "inconsistent valuation"


-- Initialize the producers and the accumulators of the consumers
-- with the given array enviroment.
initialiseSeq :: forall aenv arrs'.
                 SeqConfig
              -> ExecOpenSeq Void aenv () arrs'
              -> Aval aenv
              -> Stream
              -> CIO (StreamDAG () arrs', Int)
initialiseSeq conf topSeq aenv stream = do  
  s0 <- evalShapes topSeq
  let pd = maxStepSize (chunkSize conf) s0
  s1 <- initSeq s0 pd
  return (s1, pd)

  where
    evalClosedExp :: ExecExp () aenv t -> CIO t
    evalClosedExp exp = executeExp exp aenv stream

    initSeq :: ExecOpenSeq Id aenv senv a -> Int -> CIO (StreamDAG senv a)
    initSeq s pd =
      case s of
        ExecProducer (Id sh) p s' -> do
          (p', c) <- initProducer pd sh p
          s'' <- initSeq s' pd
          return $ StreamProducer c p' s''
        ExecConsumer   c     -> StreamConsumer <$> initConsumer c pd
        ExecReify      ix    -> return (StreamReify ix)

    initProducer :: forall sh e senv.
                    Int
                 -> sh
                 -> ExecProducer aenv senv (Array sh e)
                 -> CIO (StreamProducer senv (Array sh e), PrimChunk sh e)
    initProducer pd sh p =
      case p of
        ExecStreamIn _ arrs -> 
          do c <- allocateChunk pd sh
             return $ (StreamStreamIn arrs, c)
        ExecToSeq kernel gamma _ _ _ n -> 
          do c <- allocateChunk pd sh
             return $ (StreamMapFin n $ \ senv i s ->
                        do -- The events in aenv lives in potentially
                           -- destroyed streams.
                           let k = senvSize senv
--                           ev <- join $ liftIO <$> (Event.waypoint <$> asks activeContext <*> gets eventTable <*> pure s)
                           execute kernel gamma aenv mempty (Sempty, 1) (k * size sh) (i, asArray (setPD k c)) s
--                           peekListChunk (setPD k c)
                       , c)    
        ExecUseLazy slix _ arr@(Array sh0 _) -> 
          do c <- allocateChunk pd sh
             let n = R.size sh0 `div` (size sh `max` 1)
             return ( StreamMapFin n $ \ senv i s -> 
                        do let 
                             k = senvSize senv
                             args = copyArgs slix sh0 i (i + k)
                           mapM_ (\ x -> pokeCopyArgs x arr c) args -- FMMA TODO async?
                           -- FMMA TODO permute
                    , c)
        ExecSeqOp kernels gamma sgamma op -> 
          do c <- allocateChunk pd sh
             op' <- initSeqOp c kernels gamma sgamma op
             return (op', c)
        ExecScanSeq kernels gamma e0
          | FL () kernel more <- kernels
          , Cons _ !upsweep1 (Cons _ !upsweep2 _) <- more
          ->
          do e0' <- evalClosedExp e0
             let (_,!numIntervals,_) = configure kernel pd
             c@(Array _ adata) <- newArray      (Z :. pd + 1) (\ _ -> e0')
             blk               <- allocateArray (Z :. numIntervals) :: CIO (Vector e)
             withDevicePtrs adata (Just stream) $ \out -> do
               let (!body, !sum) = (out, advancePtrsOfArrayData adata pd out)
                   !d_body = marshalDevicePtrs adata body
                   !d_sum  = marshalDevicePtrs adata sum
                   scanner senv _ stream =
                     do let 
                          numElements = senvSize senv
                          (_,!numIntervals,_) = configure kernel numElements
                        when (numIntervals > 1) $ do
                          execute upsweep1 gamma aenv mempty senv numElements blk stream
                          execute upsweep2 gamma aenv mempty senv numIntervals (blk, blk, d_sum) stream
                        execute kernel gamma aenv mempty senv numElements (numElements, d_body, blk, d_sum) stream
               return (StreamMap scanner, PrimChunk pd Z adata)
          | otherwise -> $internalError "initProducer" "scan-seq does not have the required kernels"

    initSeqOp :: forall sh e senv a. (Shape sh, Elt e) 
              => PrimChunk sh e
              -> FullList () (AccKernel a)
              -> Gamma aenv
              -> Gamma senv
              -> PreOpenArrayOp (Idx senv) ExecOpenAcc senv aenv (Array sh e)
              -> CIO (StreamProducer senv (Array sh e))
    initSeqOp c (FL () kernel more) gamma sgamma op =
      case op of
        Map _ x                   -> return $ StreamMap $ \ senv _ s -> after s (prj' x senv) >> executeOp senv s
        Generate _ _              -> return $ StreamMap $ \ senv _ s -> executeOp senv s
        Transform _ _ _ x         -> return $ StreamMap $ \ senv _ s -> after s (prj' x senv) >> executeOp senv s
        Backpermute _ _ x         -> return $ StreamMap $ \ senv _ s -> after s (prj' x senv) >> executeOp senv s

        ZipWith _ x y             -> return $ StreamMap $ \ senv _ s -> after s (prj' x senv) >> after s (prj' y senv) >> executeOp senv s
        Replicate _ _ x           -> return $ StreamMap $ \ senv _ s -> after s (prj' x senv) >> executeOp senv s
        Slice _ x _               -> return $ StreamMap $ \ senv _ s -> after s (prj' x senv) >> executeOp senv s

        -- Consumers
        -- ---------
        
        -- Since we always have a streaming dimension, we know that
        -- (dim sh > 0) and can simplify kernel execution for folds.
        Fold _ _ x                -> return $ StreamMap $ \ senv _ s -> after s (prj' x senv) >> executeOp senv s
        Fold1 _ x                 -> return $ StreamMap $ \ senv _ s -> after s (prj' x senv) >> executeOp senv s
        
        -- FMMA TODO: Segmented folds are currently broken.
        FoldSeg _ _ x y           -> return $ StreamMap $ \ senv _ s -> foldSegOp c senv x y s -- after s (prj' x senv) >> after s (prj' y senv) >> executeOp senv s
        Fold1Seg _ x y            -> return $ StreamMap $ \ senv _ s -> foldSegOp c senv x y s -- after s (prj' x senv) >> after s (prj' y senv) >> executeOp senv s
      {-        Scanl f z x               -> scanlSop (evalClosedFun f) (evalClosedExp z) (prj' x senv) s
        Scanl1 f x                -> scanl1Sop (evalClosedFun f) (prj' x senv) s
        Scanr f z x               -> scanrSop (evalClosedFun f) (evalClosedExp z) (prj' x senv) s
        Scanr1 f x                -> scanr1Sop (evalClosedFun f) (prj' x senv) s -}
        Permute _ x _ y           -> permuteOp c y x
-- return $ StreamMap $ \ senv _ s -> after s (prj' x senv) >> after s (prj' y senv) >> permuteOp senv s
      {-  Stencil sten b x          -> stencilSop (evalClosedFun sten) b (prj' x senv)
        Stencil2 sten b1 x1 b2 x2 -> stencil2Sop (evalClosedFun sten) b1 b2 (prj' x1 senv) (prj' x2 senv)
-}
        op -> $internalError "initSeqOp" ("Operation is not defined yet:" ++ showPreArrOp op)
      where
        executeOp :: Val' senv -> Stream -> CIO ()
        executeOp senv stream = do
          execute kernel gamma aenv sgamma senv (size (chunkEltShape c) * (senvSize senv)) (asArray (setPD (senvSize senv) c)) stream

        foldSegOp :: forall sh e i. (Shape sh, Elt e, Elt i) => PrimChunk (sh :. Int) e -> Val' senv -> Idx senv (Array (sh :. Int) e) -> Idx senv (Array DIM1 i) -> Stream -> CIO ()
        foldSegOp c senv x y stream = do
          -- Marshall the arguments as flattened arrays:
          a <- asFlat <$> after stream (prj' x senv)
          s <- asFlat <$> after stream (prj' y senv)
          let sh :. _ = shape a
              Z :. sz = shape s
              c' = asArray' (setPD (senvSize senv) c)
          error "Sequence-op not implented yet: foldSeg, fold1Seg" -- TODO, scan segments
          execute kernel gamma aenv mempty (Sempty, 1) (size sh * (sz - 1)) (s, a, 1 :: Int, c') stream

        permuteOp :: forall sh sh' e. (Shape sh, Shape sh', Elt e) 
                  => PrimChunk sh' e
                  -> Idx senv (Array sh e)
                  -> Idx senv (Array sh' e)
                  -> CIO (StreamProducer senv (Array sh' e))
        permuteOp c a d = do
          let sh' = shape (asArray c)
              n'  = size sh'
          PrimChunk _ _ locks <- allocateChunk (chunkLength c) (chunkEltShape c) :: CIO (PrimChunk sh' Int32)
          withDevicePtrs locks (Just stream) $ \d_locks -> do
            liftIO $ CUDA.memsetAsync d_locks n' 0 (Just stream)
            return $ 
              StreamMap $
                \ senv _ s -> do
                  dfs <- after s (prj' d senv)
                  a <- after s (prj' a senv)
                  copyArrayAsync (asArray dfs) (asArray (setPD (senvSize senv) c)) (Just s)
                  execute kernel gamma aenv sgamma senv (senvSize senv * size (chunkEltShape a)) (asArray (setPD (senvSize senv) c), d_locks) s

    initConsumer :: forall a senv.
                    ExecConsumer aenv senv a
                 -> Int
                 -> CIO (StreamConsumer senv a)
    initConsumer c pd =
      case c of
        ExecFoldSeq kzip fin gamma (x :: Idx senv (Scalar e)) z ->
          do z' <- evalSh z
             c :: PrimChunk Z e <- newChunk pd Z (const z')
             let consumer :: Val' senv -> PrimChunk Z e -> Stream -> CIO (PrimChunk Z e)
                 consumer senv c s =
                   do let c' = setPD (senvSize senv) c
                      d  <- after s (prj' x senv)
                      execute kzip gamma aenv mempty (Sempty, 1) (senvSize senv) (asArray c', asArray d) s
                      return c
                 finalizer :: PrimChunk Z e -> Stream -> CIO (Scalar e)
                 finalizer c _ = 
                   do nop <- join $ liftIO <$> (Event.create <$> asks activeContext <*> gets eventTable)
                      executeOpenAfun1 fin aenv (Async nop (asArray c))
             return $ StreamFold consumer finalizer c
        ExecFoldSeqFlatten f acc x ->
          do a0 <- executeOpenAcc acc aenv stream
             let consumer senv a _ =
                   let c = prj' x senv
                   in executeOpenAfun2 f aenv (fmap (const a) c) (fmap asArray c)
             let finalizer x _ = return x
             return $ StreamFold consumer finalizer a0
        ExecStuple t ->
          let initTup :: Atuple (ExecConsumer aenv senv) t -> CIO (Atuple (StreamConsumer senv) t)
              initTup NilAtup        = return NilAtup
              initTup (SnocAtup t c) = SnocAtup <$> initTup t <*> initConsumer c pd
          in StreamStuple <$> initTup t

    evalSh :: ExecExp () aenv sh -> CIO sh
    evalSh sh = executeExp sh aenv stream

    evalShapes :: ExecOpenSeq Void aenv () a -> CIO (ExecOpenSeq Id aenv () a)
    evalShapes = flip go EmptyShapes
      where
        go :: ExecOpenSeq Void aenv senv a -> Shapes senv -> CIO (ExecOpenSeq Id aenv senv a)
        go (ExecProducer _ (ExecToSeq k gamma slice slix shExp _) s) shs =
          do sh <- evalSh shExp
             let sl = sliceShape slice sh
                 n = size sh `div` (size sl `max` 1)
             s' <- go s (shs `PushShape` sl)
             return (ExecProducer (Id sl) (ExecToSeq k gamma slice slix shExp n) s')
        go (ExecProducer _ p s) shs =
          do sh <- producerShape p shs
             s' <- go s (shs `PushShape` sh)
             return $ ExecProducer (Id sh) p s'
        go (ExecConsumer c) _ = return $ ExecConsumer c
        go (ExecReify x)    _ = return $ ExecReify x

        producerShape :: Shape sh => ExecProducer aenv senv (Array sh e) -> Shapes senv -> CIO sh
        producerShape p shs =
          case p of
            ExecStreamIn sh _ -> evalSh sh
            ExecUseLazy slice slix arr -> return $ sliceShape slice (shape arr)
            ExecSeqOp _ _ _ op -> seqOpShape op shs
            ExecScanSeq _ _ _ -> return $ Z
            ExecToSeq{} -> error "Already handled in evalShapes"

        seqOpShape :: Shape sh => PreOpenArrayOp (Idx senv) ExecOpenAcc senv aenv (Array sh e) -> Shapes senv -> CIO sh
        seqOpShape op shs =
          case op of
            Map _ x                   -> return $ prjShape x shs
            Generate sh _             -> evalSh sh
            Transform sh _ _ _        -> evalSh sh
            Backpermute sh _ _        -> evalSh sh

            ZipWith _ x1 x2           -> return $ prjShape x1 shs `intersect` prjShape x2 shs
            Replicate slice slix x    -> do sl <- evalSh slix; return $ toElt $ fst $ extend slice (fromElt sl) (fromElt (prjShape x shs))
            Slice slice x slix        -> do sl <- evalSh slix; return $ toElt $ fst $ restrict slice (fromElt sl) (fromElt (prjShape x shs))

            -- Consumers
            -- ---------
            Fold _ _ x                -> return $ let sh :. _ = prjShape x shs in (listToShape . map (max 1) . shapeToList $ sh)
            Fold1 _ x                 -> return $ let sh :. _ = prjShape x shs in (listToShape . map (max 1) . shapeToList $ sh)
            FoldSeg _ _ x y           -> return $ let sh :. _ = prjShape x shs in let Z :. sz = prjShape y shs in sh :. sz
            Fold1Seg _ x y            -> return $ let sh :. _ = prjShape x shs in let Z :. sz = prjShape y shs in sh :. sz
            Scanl _ _ x               -> return $ let Z :. n = prjShape x shs in Z :. n + 1
            Scanl1 _ x                -> return $ prjShape x shs
            Scanr _ _ x               -> return $ let Z :. n = prjShape x shs in Z :. n + 1
            Scanr1 _ x                -> return $ prjShape x shs
            Permute _ x1 _ _          -> return $ prjShape x1 shs
            Stencil _ _ x             -> return $ prjShape x shs
            Stencil2 _ _ x1 _ x2      -> return $ prjShape x1 shs `intersect` prjShape x2 shs

    maxStepSize :: Int -> ExecOpenSeq Id aenv () a -> Int
    maxStepSize maxChunkSize s =
      let elemSize = maxEltSize s
          (a,b) = maxChunkSize `quotRem` (elemSize `max` 1)
      in stepSize' (a + signum b) s
      where
        maxEltSize :: ExecOpenSeq Id aenv senv a -> Int
        maxEltSize (ExecProducer (Id sh) _ s0) = size sh `max` maxEltSize s0
        maxEltSize _ = 0

        stepSize' :: Int -> ExecOpenSeq Id aenv senv a -> Int
        stepSize' n s =
          case s of
            ExecProducer _ p s0 -> min (stepSize' n s0) $
              case p of
                ExecStreamIn _ xs -> length (take n xs)
                ExecToSeq  _ _ _ _ _ k -> n `min` k
                _ -> n
            _ -> n

executeSequence :: forall aenv arrs . Arrays arrs
                => SeqConfig
                -> ExecOpenSeq Void aenv () arrs
                -> Aval aenv
                -> Stream
                -> CIO arrs
executeSequence conf s aenv stream
  = evalSeq'
  where
    evalSeq' :: CIO arrs
    evalSeq' =
      do (s1, pd) <- initialiseSeq conf s aenv stream
         loop pd 0 s1

    -- Iterate the given sequence until it terminates.
    -- A sequence only terminates when one of the producers are exhausted.
    loop :: Arrays arrs
         => Int
         -> Int
         -> StreamDAG () arrs
         -> CIO arrs
    loop n i s =
      do let k = stepSize n s
         if k == 0
           then returnOut s
           else loop n (i + k) =<< step s i (Sempty, k)

    returnOut :: forall senv. StreamDAG senv arrs -> CIO arrs
    returnOut s = 
      case s of
        StreamProducer c _ s0 -> {- freeArray (asArray c) >> -} returnOut s0
        StreamReify _ -> error "Absurd"
        StreamConsumer c -> retC c
        where 
          retC :: StreamConsumer senv a -> CIO a
          retC c =
            case c of
              StreamFold _ f x -> streaming (f x) wait
              StreamStuple t -> 
                let retT :: forall t. Atuple (StreamConsumer senv) t -> CIO t
                    retT NilAtup          = return ()
                    retT (SnocAtup t0 c0) = (,) <$> retT t0 <*> retC c0
                in toAtuple <$> retT t

    stepSize :: Int -> StreamDAG senv arrs' -> Int
    stepSize n s =
      case s of
        StreamProducer _ p s0 -> min (stepSize n s0) $
          case p of
            StreamStreamIn xs -> length (take n xs)
            StreamMapFin k _ -> n `min` k
            _ -> n
        _ -> n

    -- One iteration of a stream.
    step :: forall senv arrs'.
            StreamDAG senv arrs'
         -> Int
         -> Val' senv
         -> CIO (StreamDAG senv arrs')
    step s i senv =
      case s of
        StreamProducer c p s' ->
          do let c' = setPD (senvSize senv) c
             streaming (produce c' p senv i)
               $ \ (Async ev p') -> do
                 s'' <- step s' i (senv `push'` Async ev c')
                 return (StreamProducer c' p' s'')
        StreamConsumer c -> StreamConsumer <$> consume c senv
        StreamReify _ -> $internalError "step" "Absurd"

produce :: forall sh e senv. (Shape sh, Elt e)
        => PrimChunk sh e
        -> StreamProducer senv (Array sh e)
        -> Val' senv
        -> Int
        -> Stream
        -> CIO (StreamProducer senv (Array sh e))
produce c p senv i stream =
  case p of
    StreamStreamIn xs ->
      do let k           = senvSize senv
             sh          = chunkEltShape c
             (xs', xs'') = (take k xs, drop k xs)
         _ <- mapM (\ (a, i) -> pokeArrayInto a c i) (zip xs' [0..])  -- FMMA TODO async?
         return (StreamStreamIn xs'')
    StreamMap f ->
      do () <- streaming (f senv i) wait
         return (StreamMap f)

    StreamMapFin n f ->
      do () <- streaming (f senv i) wait
         return (StreamMapFin (n - senvSize senv) f)

consume :: forall senv a. StreamConsumer senv a -> Val' senv -> CIO (StreamConsumer senv a)
consume c senv =
  case c of
    StreamFold f g acc ->
      do acc' <- streaming (f senv acc) wait
         return (StreamFold f g acc')
    StreamStuple t ->
      let consT :: Atuple (StreamConsumer senv) t -> CIO (Atuple (StreamConsumer senv) t)
          consT NilAtup        = return NilAtup
          consT (SnocAtup t0 c0) = SnocAtup <$> consT t0 <*> consume c0 senv
      in StreamStuple <$> consT t

-- Evaluating bindings
-- -------------------

executeExtend :: Extend ExecOpenAcc aenv aenv' -> Aval aenv -> (Aval aenv' -> CIO () -> CIO r) -> CIO r
executeExtend BaseEnv       aenv c = c aenv (return ())
executeExtend (PushEnv e a) aenv c =
  executeExtend e aenv (\ aenv' free1 -> streaming' (executeOpenAcc a aenv') $ \a' free2 -> c (aenv' `Apush` a') (free1 >> free2))

-- Scalar expression evaluation
-- ----------------------------

executeExp :: ExecExp () aenv t -> Aval aenv -> Stream -> CIO t
executeExp !exp !aenv !stream = executeOpenExp exp Empty aenv stream

executeOpenExp :: forall env aenv exp. ExecOpenExp env () aenv exp -> Val env -> Aval aenv -> Stream -> CIO exp
executeOpenExp !rootExp !env !aenv !stream = travE rootExp
  where
    travE :: ExecOpenExp env () aenv t -> CIO t
    travE exp = case exp of
      Var ix                    -> return (prj ix env)
      Let bnd body              -> travE bnd >>= \x -> executeOpenExp body (env `Push` x) aenv stream
      Const c                   -> return (toElt c)
      PrimConst c               -> return (evalPrimConst c)
      PrimApp f x               -> evalPrim f <$> travE x
      Tuple t                   -> toTuple <$> travT t
      Prj ix e                  -> evalPrj ix . fromTuple <$> travE e
      Cond p t e                -> travE p >>= \x -> if x then travE t else travE e
      While p f x               -> while p f =<< travE x
      IndexAny                  -> return Any
      IndexNil                  -> return Z
      IndexCons sh sz           -> (:.) <$> travE sh <*> travE sz
      IndexHead sh              -> (\(_  :. ix) -> ix) <$> travE sh
      IndexTail sh              -> (\(ix :.  _) -> ix) <$> travE sh
      IndexSlice ix slix sh     -> indexSlice ix <$> travE slix <*> travE sh
      IndexFull ix slix sl      -> indexFull  ix <$> travE slix <*> travE sl
      ToIndex sh ix             -> toIndex   <$> travE sh  <*> travE ix
      FromIndex sh ix           -> fromIndex <$> travE sh  <*> travE ix
      Intersect sh1 sh2         -> intersect <$> travE sh1 <*> travE sh2
      Union sh1 sh2             -> union <$> travE sh1 <*> travE sh2
      ShapeSize sh              -> size  <$> travE sh
      Shape acc                 -> shape <$> travA acc
      Index acc ix              -> join $ index      <$> travA acc <*> travE ix
      LinearIndex acc ix        -> join $ indexArray <$> travA acc <*> travE ix
      ShapeS{}                  -> error "Absurd"
      IndexS{}                  -> error "Absurd"
      LinearIndexS{}            -> error "Absurd"
      Foreign _ f x             -> foreign f x

    -- Helpers
    -- -------

    travT :: Tuple (ExecOpenExp env () aenv) t -> CIO t
    travT tup = case tup of
      NilTup            -> return ()
      SnocTup !t !e     -> (,) <$> travT t <*> travE e

    travA :: ExecOpenAcc aenv a -> CIO a
    travA !acc = do
      executeOpenAcc acc aenv stream

    foreign :: ExecFun () () (a -> b) -> ExecOpenExp env () aenv a -> CIO b
    foreign (Lam (Body f)) x = travE x >>= \e -> executeOpenExp f (Empty `Push` e) Aempty stream
    foreign _              _ = error "I bless the rains down in Africa"

    travF1 :: ExecOpenFun env () aenv (a -> b) -> a -> CIO b
    travF1 (Lam (Body f)) x = executeOpenExp f (env `Push` x) aenv stream
    travF1 _              _ = error "Gonna take some time to do the things we never have"

    while :: ExecOpenFun env () aenv (a -> Bool) -> ExecOpenFun env () aenv (a -> a) -> a -> CIO a
    while !p !f !x = do
      ok <- travF1 p x
      if ok then while p f =<< travF1 f x
            else return x

    indexSlice :: (Elt slix, Elt sh, Elt sl)
               => SliceIndex (EltRepr slix) (EltRepr sl) co (EltRepr sh)
               -> slix
               -> sh
               -> sl
    indexSlice !ix !slix !sh = toElt $! restrict ix (fromElt slix) (fromElt sh)
      where
        restrict :: SliceIndex slix sl co sh -> slix -> sh -> sl
        restrict SliceNil              ()        ()       = ()
        restrict (SliceAll   sliceIdx) (slx, ()) (sl, sz) = (restrict sliceIdx slx sl, sz)
        restrict (SliceFixed sliceIdx) (slx,  _) (sl,  _) = restrict sliceIdx slx sl

    indexFull :: (Elt slix, Elt sh, Elt sl)
              => SliceIndex (EltRepr slix) (EltRepr sl) co (EltRepr sh)
              -> slix
              -> sl
              -> sh
    indexFull !ix !slix !sl = toElt $! extend ix (fromElt slix) (fromElt sl)
      where
        extend :: SliceIndex slix sl co sh -> slix -> sl -> sh
        extend SliceNil              ()        ()       = ()
        extend (SliceAll sliceIdx)   (slx, ()) (sh, sz) = (extend sliceIdx slx sh, sz)
        extend (SliceFixed sliceIdx) (slx, sz) sh       = (extend sliceIdx slx sh, sz)

    index :: (Shape sh, Elt e) => Array sh e -> sh -> CIO e
    index !arr !ix = indexArray arr (toIndex (shape arr) ix)

extend :: SliceIndex slix sl co dim
       -> slix
       -> sl
       -> (dim, dim -> sl)
extend SliceNil              ()        ()       = ((), const ())
extend (SliceAll sliceIdx)   (slx, ()) (sl, sz)
  = let (dim', f') = extend sliceIdx slx sl
    in  ((dim', sz), \(ix, i) -> (f' ix, i))
extend (SliceFixed sliceIdx) (slx, sz) sl
  = let (dim', f') = extend sliceIdx slx sl
    in  ((dim', sz), \(ix, _) -> f' ix)

restrict :: SliceIndex slix sl co sh
         -> slix
         -> sh
         -> (sl, sl -> sh)
restrict SliceNil              ()        ()       = ((), const ())
restrict (SliceAll sliceIdx)   (slx, ()) (sl, sz)
  = let (sl', f') = restrict sliceIdx slx sl
    in  ((sl', sz), \(ix, i) -> (f' ix, i))
restrict (SliceFixed sliceIdx) (slx, i)  (sl, sz)
  = let (sl', f') = restrict sliceIdx slx sl
    in  $indexCheck "slice" i sz $ (sl', \ix -> (f' ix, i))

-- Marshalling data
-- ----------------

marshalSlice' :: SliceIndex slix sl co dim
              -> slix
              -> CIO [CUDA.FunParam]
marshalSlice' SliceNil () = return []
marshalSlice' (SliceAll sl)   (sh, ()) = marshalSlice' sl sh
marshalSlice' (SliceFixed sl) (sh, n)  =
  do x  <- evalContT $ marshal n Nothing
     xs <- marshalSlice' sl sh
     return (xs ++ x)

marshalSlice :: Elt slix => SliceIndex (EltRepr slix) sl co dim
             -> slix
             -> CIO [CUDA.FunParam]
marshalSlice slix = marshalSlice' slix . fromElt

-- Data which can be marshalled as function arguments to a kernel invocation.
--
class Marshalable a where
  marshal :: a -> Maybe Stream -> ContT b CIO [CUDA.FunParam]

instance Marshalable () where
  marshal () _ = return []

instance Marshalable CUDA.FunParam where
  marshal !x _ = return [x]

instance ArrayElt e => Marshalable (ArrayData e) where
  marshal !ad ms = ContT $ marshalArrayData ad ms

instance Shape sh => Marshalable sh where
  marshal !sh ms = marshal (reverse (shapeToList sh)) ms

instance Marshalable a => Marshalable [a] where
  marshal xs ms = concatMapM (flip marshal ms) xs

instance (Marshalable sh, Elt e) => Marshalable (Array sh e) where
  marshal !(Array sh ad) ms = (++) <$> marshal (toElt sh :: sh) ms <*> marshal ad ms

instance (Marshalable a, Marshalable b) => Marshalable (a, b) where
  marshal (!a, !b) ms = (++) <$> marshal a ms <*> marshal b ms

instance (Marshalable a, Marshalable b, Marshalable c) => Marshalable (a, b, c) where
  marshal (!a, !b, !c) ms
    = concat <$> sequence [marshal a ms, marshal b ms, marshal c ms]

instance (Marshalable a, Marshalable b, Marshalable c, Marshalable d)
      => Marshalable (a, b, c, d) where
  marshal (!a, !b, !c, !d) ms
    = concat <$> sequence [marshal a ms, marshal b ms, marshal c ms, marshal d ms]

#define primMarshalable(ty)                                                    \
instance Marshalable (ty) where {                                              \
  marshal !x _ = return [CUDA.VArg x] }

primMarshalable(Int)
primMarshalable(Int8)
primMarshalable(Int16)
primMarshalable(Int32)
primMarshalable(Int64)
primMarshalable(Word)
primMarshalable(Word8)
primMarshalable(Word16)
primMarshalable(Word32)
primMarshalable(Word64)
primMarshalable(Float)
primMarshalable(Double)
primMarshalable(CUDA.DevicePtr a)


-- Note [Array references in scalar code]
-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
--
-- All CUDA devices have between 6-8KB of read-only texture memory per
-- multiprocessor. Since all arrays in Accelerate are immutable, we can always
-- access input arrays through the texture cache to reduce global memory demand
-- when accesses do not follow the regular patterns required for coalescing.
--
-- This is great for older 1.x series devices, but newer devices have a
-- dedicated L2 cache (device dependent, 256KB-1.5MB), as well as a configurable
-- L1 cache combined with shared memory (16-48KB).
--
-- For older 1.x series devices, we pass free array variables as texture
-- references, but for new devices we pass them as standard array arguments so
-- as to use the larger available caches.
--

marshalAccEnvTex :: AccKernel a -> Aval aenv -> Gamma aenv -> Val' senv -> Gamma senv -> Stream -> ContT b CIO [CUDA.FunParam]
marshalAccEnvTex !kernel !aenv (Gamma !gamma) !senv (Gamma !sgamma) !stream
  = flip concatMapM (Map.toList gamma)
  $ \(Idx_ !(idx :: Idx aenv (Array sh e)), i) ->
        do arr <- after stream (aprj idx aenv)
           ContT $ \f -> marshalAccTex (namesOfArray (groupOfInt i) (undefined :: e)) kernel arr (Just stream) (f ())
           marshal (shape arr) (Just stream)

marshalAccTex :: (Name,[Name]) -> AccKernel a -> Array sh e -> Maybe Stream -> CIO b -> CIO b
marshalAccTex (_, !arrIn) (AccKernel _ _ !lmdl _ _ _ _) (Array !sh !adata) ms run
  = do
      texs <- liftIO $ withLifetime lmdl $ \mdl -> (sequence' $ map (CUDA.getTex mdl) (reverse arrIn))
      marshalTextureData adata (R.size sh) texs ms (const run)

marshalAccEnvArg :: Aval aenv -> Gamma aenv -> Val' senv -> Gamma senv -> Stream -> ContT b CIO [CUDA.FunParam]
marshalAccEnvArg !aenv (Gamma !gamma) (!senv, k) (Gamma !sgamma) !stream
  = do xs <- concatMapM (\(Idx_ !idx) -> flip marshal (Just stream) =<< after stream (aprj idx aenv)) (Map.keys gamma)
       ys <- concatMapM (\(Idx_ !idx) -> flip marshal (Just stream) . asArray' =<< after stream (sprj idx senv)) (Map.keys sgamma)
       k' <- marshal k (Just stream)
       return $ xs ++ ys ++ concat [k' | Map.size sgamma > 0] -- Only marshal the chunk size if there are any chunks..

-- A lazier version of 'Control.Monad.sequence'
--
sequence' :: [IO a] -> IO [a]
sequence' = foldr k (return [])
  where k m ms = do { x <- m; xs <- unsafeInterleaveIO ms; return (x:xs) }

-- Generalise concatMap for teh monadz
--
concatMapM :: Monad m => (a -> m [b]) -> [a] -> m [b]
concatMapM f xs = concat `liftM` mapM f xs

-- Kernel execution
-- ----------------

-- What launch parameters should we use to execute the kernel with a number of
-- array elements?
--
configure :: AccKernel a -> Int -> (Int, Int, Int)
configure (AccKernel _ _ _ _ !cta !smem !grid) !n = (cta, grid n, smem)


-- Marshal the kernel arguments. For older 1.x devices this binds free arrays to
-- texture references, and for newer devices adds the parameters to the front of
-- the argument list
--
arguments :: Marshalable args
          => AccKernel a
          -> Aval aenv
          -> Gamma aenv
          -> Val' senv
          -> Gamma senv
          -> args
          -> Stream
          -> ContT b CIO [CUDA.FunParam]
arguments !kernel !aenv !gamma !senv !sgamma !a !stream = do
  dev <- asks deviceProperties
  let marshaller | computeCapability dev < Compute 2 0   = marshalAccEnvTex kernel
                 | otherwise                             = marshalAccEnvArg
  --
  (++) <$> marshaller aenv gamma senv sgamma stream <*> marshal a (Just stream)


-- Link the binary object implementing the computation, configure the kernel
-- launch parameters, and initiate the computation. This also handles lifting
-- and binding of array references from scalar expressions.
--
execute :: Marshalable args
        => AccKernel a                  -- The binary module implementing this kernel
        -> Gamma aenv                   -- variables of arrays embedded in scalar expressions
        -> Aval aenv                    -- the environment
        -> Gamma senv                   -- variables of arrays embedded in scalar expressions
        -> Val' senv                    -- the environment
        -> Int                          -- a "size" parameter, typically number of elements in the output
        -> args                         -- arguments to marshal to the kernel function
        -> Stream                       -- Compute stream to execute in
        -> CIO ()
execute !kernel !gamma !aenv !sgamma !senv !n !a !stream = evalContT $ do
  args  <- arguments kernel aenv gamma senv sgamma a stream
  liftIO $ launch kernel (configure kernel n) args stream


-- Execute a device function, with the given thread configuration and function
-- parameters. The tuple contains (threads per block, grid size, shared memory)
--
launch :: AccKernel a -> (Int,Int,Int) -> [CUDA.FunParam] -> Stream -> IO ()
launch (AccKernel entry !fn _ _ _ _ _) !(cta, grid, smem) !args !stream
  = D.timed D.dump_exec msg (Just stream)
  $ CUDA.launchKernel fn (grid,1,1) (cta,1,1) smem (Just stream) args
  where
    msg gpuTime cpuTime
      = "exec: " ++ entry ++ "<<< " ++ shows grid ", " ++ shows cta ", " ++ shows smem " >>> "
                 ++ D.elapsed gpuTime cpuTime


streamSeq :: Arrays a => ExecSeq [a] -> CIO (LazySeq a)
streamSeq (ExecS !ext !s) = streaming go wait
  where
    go !stream = do
      executeExtend ext Aempty
        (\ aenv freeStreams -> 
          streamOutSeq defaultSeqConfig s aenv freeStreams stream)

streamOutSeq :: forall aenv arrs . Arrays arrs
             => SeqConfig
             -> ExecOpenSeq Void aenv () [arrs]
             -> Aval aenv
             -> CIO () -- Stream destroyer for the streams producing aenv.
             -> Stream
             -> CIO (LazySeq arrs)
streamOutSeq conf s aenv freeStreams stream
  = evalSeq'
  where
    evalSeq' :: CIO (LazySeq arrs)
    evalSeq' =
      do (s1, pd) <- initialiseSeq conf s aenv stream
         loop pd s1 0

    -- Iterate the given sequence until it terminates.
    -- A sequence only terminates when one of the producers are exhausted.
    loop :: Arrays arrs
         => Int
         -> StreamDAG () [arrs]
         -> Int
         -> CIO (LazySeq arrs)
    loop n s i =
      do let k = stepSize n s
         if k == 0
           then do
             free s
             freeStreams -- Release the streams for aenv. FMMA TODO maybe free aenv explicitely as well?
             return Done
           else do     
             (as, s') <- step s (Sempty, k) i
             return $ Yield as (loop n s' (i + k))

    stepSize :: Int -> StreamDAG senv arrs' -> Int
    stepSize n s =
      case s of
        StreamProducer _ p s0 -> min (stepSize n s0) $
          case p of
            StreamStreamIn xs -> length (take n xs)
            StreamMapFin k _ -> n `min` k
            _ -> n
        _ -> n

    free :: forall senv arrs'. StreamDAG senv arrs'
         -> CIO ()
    free s =
      case s of
        StreamProducer c _ s0 -> {- freeArray (asArray c) >> -} free s0
        StreamReify _ -> return ()
        StreamConsumer _ -> error "Absurd"

    -- One iteration of a stream.
    step :: forall senv arrs'.
            StreamDAG senv arrs'
         -> Val' senv
         -> Int
         -> CIO (arrs', StreamDAG senv arrs')
    step s senv i =
      case s of
        StreamProducer c p s' ->
          do let c' = setPD (senvSize senv) c
             streaming (produce c' p senv i)
               $ \ (Async ev p') -> do
                 (x, s'') <- step s' (senv `push'` Async ev c') i
                 return (x, StreamProducer c' p' s'')
        StreamReify x -> 
          do as <- peekArraysOfChunk =<< wait (prj' x senv) -- FMMA TODO async?
             return (as, StreamReify x)
        StreamConsumer c -> error "Absurd"


-- FMMA TODO Use generate or memset instead, move to sugar
newChunk :: (Shape sh, Elt e) => Int -> sh -> (sh -> e) -> CIO (PrimChunk sh e)
newChunk pd sh f = 
  do Array _ adata <- newArray (pd .: sh) (f . indexInit)
     return (PrimChunk pd sh adata)

