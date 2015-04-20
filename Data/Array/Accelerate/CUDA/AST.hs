{-# LANGUAGE EmptyDataDecls             #-}
{-# LANGUAGE FlexibleContexts           #-}
{-# LANGUAGE FlexibleInstances          #-}
{-# LANGUAGE GADTs                      #-}
{-# LANGUAGE GeneralizedNewtypeDeriving #-}
{-# LANGUAGE KindSignatures             #-}
{-# LANGUAGE TemplateHaskell            #-}
{-# LANGUAGE TypeOperators              #-}
{-# LANGUAGE TypeSynonymInstances       #-}
-- |
-- Module      : Data.Array.Accelerate.CUDA.AST
-- Copyright   : [2008..2014] Manuel M T Chakravarty, Gabriele Keller
--               [2009..2014] Trevor L. McDonell
-- License     : BSD3
--
-- Maintainer  : Trevor L. McDonell <tmcdonell@cse.unsw.edu.au>
-- Stability   : experimental
-- Portability : non-portable (GHC extensions)
--

module Data.Array.Accelerate.CUDA.AST (

  module Data.Array.Accelerate.AST,

  AccKernel(..), Free, Gamma(..), Idx_(..),
  ExecAcc, ExecAfun, ExecOpenAfun, ExecOpenAcc(..),
  ExecExp, ExecFun, ExecOpenExp, ExecOpenFun,
  ExecSeq(..), ExecOpenSeq(..), ExecProducer(..), ExecConsumer(..),
  freevar, makeEnvMap,
  Chunk, Void(..),

) where

import Data.Array.Accelerate.CUDA.Array.Prim (DevicePtrs)
-- friends
import Data.Array.Accelerate.AST
import Data.Array.Accelerate.Lifetime
import Data.Array.Accelerate.Pretty
import Data.Array.Accelerate.Array.Sugar                ( Array, Shape, Elt, Arrays, Atuple, TupleRepr, IsAtuple, Vector, Scalar, (:.), EltRepr )
import Data.Array.Accelerate.Array.Representation       ( SliceIndex(..) )
import Data.Array.Accelerate.Trafo                      ( Extend )
import qualified Data.Array.Accelerate.FullList         as FL
import qualified Foreign.CUDA.Driver                    as CUDA
import qualified Foreign.CUDA.Analysis                  as CUDA

-- system
import Text.PrettyPrint
import Data.Hashable
import Data.Monoid                                      ( Monoid(..) )
import qualified Data.HashSet                           as Set
import qualified Data.HashMap.Strict                    as Map


-- A non-empty list of binary objects will be used to execute a kernel. We keep
-- auxiliary information together with the compiled module, such as entry point
-- and execution information.
--
data AccKernel a where
  AccKernel :: !String                                -- __global__ entry function name
            -> {-# UNPACK #-} !CUDA.Fun               -- __global__ function object
            -> {-# UNPACK #-} !(Lifetime CUDA.Module) -- binary module
            -> {-# UNPACK #-} !CUDA.Occupancy         -- occupancy analysis
            -> {-# UNPACK #-} !Int                    -- thread block size
            -> {-# UNPACK #-} !Int                    -- shared memory per block (bytes)
            -> !(Int -> Int)                          -- number of blocks for input problem size
            -> AccKernel a


-- Kernel execution is asynchronous, barriers allow (cross-stream)
-- synchronisation to determine when the operation has completed
--
-- data AccBarrier = AB !Stream !Event


-- The set of free array variables for array computations that were embedded
-- within scalar expressions. These arrays are are required to execute the
-- kernel, by binding to texture references to similar.
--
type Free aenv = Set.HashSet (Idx_ aenv)

freevar :: (Shape sh, Elt e) => Idx aenv (Array sh e) -> Free aenv
freevar = Set.singleton . Idx_


-- A mapping between environment indexes and some token identifying that array
-- in the generated code. This simply compresses the sequence of array indices
-- into a continuous range, rather than directly using the integer equivalent of
-- the de Bruijn index.
--
-- This results in generated code that is (slightly) less sensitive to the
-- placement of let bindings, ultimately leading to a higher hit rate in the
-- compilation cache.
--
newtype Gamma aenv = Gamma ( Map.HashMap (Idx_ aenv) Int )
  deriving ( Monoid )

makeEnvMap :: Free aenv -> Gamma aenv
makeEnvMap indices
  = Gamma
  . Map.fromList
  . flip zip [0..]
--  . sortBy (compare `on` idxType)
  $ Set.toList indices
--  where
--    idxType :: Idx_ aenv -> TypeRep
--    idxType (Idx_ (_ :: Idx aenv (Array sh e))) = typeOf (undefined :: e)


-- Opaque array environment indices
--
data Idx_ aenv where
  Idx_ :: (Shape sh, Elt e) => Idx aenv (Array sh e) -> Idx_ aenv

instance Eq (Idx_ aenv) where
  Idx_ ix1 == Idx_ ix2 = idxToInt ix1 == idxToInt ix2

instance Hashable (Idx_ aenv) where
  hashWithSalt salt (Idx_ ix)
    = salt `hashWithSalt` idxToInt ix

data Void a = Void

-- Interleave compilation & execution state annotations into an open array
-- computation AST
--
data ExecOpenAcc aenv a where
  ExecAcc   :: {-# UNPACK #-} !(FL.FullList () (AccKernel a))   -- executable binary objects
            -> !(Gamma aenv)                                    -- free array variables the kernel needs access to
            -> !(PreOpenAcc ExecOpenAcc aenv a)                 -- the actual computation
            -> ExecOpenAcc aenv a                               -- the recursive knot

  EmbedAcc  :: (Shape sh, Elt e)
            => !(PreExp ExecOpenAcc () aenv sh)                 -- shape of the result array, used by execution
            -> ExecOpenAcc aenv (Array sh e)

  ExecSeq   :: Arrays arrs
            => ExecOpenSeq Void aenv () arrs
            -> ExecOpenAcc aenv arrs

-- An annotated AST suitable for execution in the CUDA environment
--
type ExecAcc  a         = ExecOpenAcc () a
type ExecAfun a         = PreAfun ExecOpenAcc a
type ExecOpenAfun aenv a = PreOpenAfun ExecOpenAcc aenv a

type ExecOpenExp        = PreOpenExp ExecOpenAcc
type ExecOpenFun        = PreOpenFun ExecOpenAcc

type ExecExp            = ExecOpenExp ()
type ExecFun            = ExecOpenFun ()


-- Display the annotated AST
-- -------------------------

instance Show (ExecOpenAcc aenv a) where
  show = render . prettyExecAcc 0 noParens

instance Show (ExecAfun a) where
  show = render . prettyExecAfun 0

prettyExecAfun :: Int -> ExecAfun a -> Doc
prettyExecAfun alvl pfun = prettyPreAfun prettyExecAcc alvl pfun

prettyExecAcc :: PrettyAcc ExecOpenAcc
prettyExecAcc alvl wrap exec =
  case exec of
    EmbedAcc sh ->
      wrap $ hang (text "Embedded") 2
           $ sep [ prettyPreExp prettyExecAcc 0 alvl parens sh ]

    ExecAcc _ (Gamma fv) pacc ->
      let base      = prettyPreAcc prettyExecAcc alvl wrap pacc
          ann       = braces (freevars (Map.keys fv))
          freevars  = (text "fv=" <>) . brackets . hcat . punctuate comma
                                      . map (\(Idx_ ix) -> char 'a' <> int (idxToInt ix))
      in
      case pacc of
        Avar{}          -> base
        Alet{}          -> base
        Apply{}         -> base
        Acond{}         -> base
        Atuple{}        -> base
        Aprj{}          -> base
        _               -> ann <+> base

    ExecSeq _ -> text "<SequenceComputation>"

data ExecSeq a where
  ExecS :: Extend ExecOpenAcc () aenv -> ExecOpenSeq Void aenv () a -> ExecSeq a

type Chunk sh e = Array (sh :. Int) e

data ExecOpenSeq shape aenv senv arrs where
  ExecProducer :: (Shape sh, Elt e)
           => shape sh
           -> ExecProducer aenv senv (Array sh e)
           -> ExecOpenSeq shape aenv (senv, Array sh e) arrs
           -> ExecOpenSeq shape aenv senv arrs

  ExecConsumer :: Arrays arrs
           => ExecConsumer aenv senv arrs
           -> ExecOpenSeq shape aenv senv arrs

  ExecReify :: (Shape sh, Elt e)
            => Idx senv (Array sh e)
            -> ExecOpenSeq shape aenv senv [Array sh e]

data ExecProducer aenv senv a where
  -- Convert the given Haskell-list of arrays to a sequence.
  ExecStreamIn :: (Shape sh, Elt e)
               => ExecExp () aenv sh
               -> [Array sh e]
               -> ExecProducer aenv senv (Array sh e)

  -- Convert the given array to a sequence.
  ExecToSeq :: (Elt slix, Shape sl, Shape sh, Elt e)
            => !(AccKernel (Chunk sl e))
            -> !(Gamma aenv)
            -> !( SliceIndex  (EltRepr slix)
                              (EltRepr sl)
                              co
                              (EltRepr sh))
            -> !(proxy slix)
            -> !(PreExp ExecOpenAcc () aenv sh)   
            -> Int -- N slices
            -> ExecProducer aenv senv (Array sl e)

  -- Convert the given array to a sequence.
  ExecUseLazy :: (Elt slix, Shape sl, Shape sh, Elt e)
              => !( SliceIndex  (EltRepr slix)
                                (EltRepr sl)
                                co
                                (EltRepr sh))
              -> !(proxy slix)
              -> Array sh e
              -> ExecProducer aenv senv (Array sl e)

  -- Map a basic array operation over a sequence.
  ExecSeqOp :: (Shape sh, Elt e)
            => {-# UNPACK #-} !(FL.FullList () (AccKernel (Chunk sh e)))
            -> !(Gamma aenv)
            -> !(Gamma senv)
            -> !(PreOpenArrayOp (Idx senv) ExecOpenAcc senv aenv (Array sh e))
            -> ExecProducer aenv senv (Array sh e)

  -- ScanSeq (+) a0 x. Scan a sequence x by combining each element
  -- using the given binary operation (+). (+) must be associative:
  --
  --   Forall a b c. (a + b) + c = a + (b + c),
  --
  -- and a0 must be the identity element for (+):
  --
  --   Forall a. a0 + a = a = a + a0.
  --
  ExecScanSeq :: Elt e
              => {-# UNPACK #-} !(FL.FullList () (AccKernel (Vector e)))
              -> !(Gamma aenv)
              -> ExecExp () aenv e
              -> ExecProducer aenv senv (Scalar e)

data ExecConsumer aenv senv a where

  -- FoldSeq (+) a0 x. Fold a sequence x by combining each element
  -- using the given binary operation (+). (+) must be associative:
  --
  --   Forall a b c. (a + b) + c = a + (b + c),
  --
  -- and a0 must be the identity element for (+):
  --
  --   Forall a. a0 + a = a = a + a0.
  --
  ExecFoldSeq :: Elt e
              => {-# UNPACK #-} !(AccKernel (Vector e)) -- Zipping kernel
              -> ExecOpenAfun aenv (Vector e -> Scalar e)
              -> !(Gamma aenv)
              -> !(Idx senv (Scalar e))
              -> ExecExp () aenv e
              -> ExecConsumer aenv senv (Scalar e)

  -- FoldSeqFlatten f a0 x. A specialized version of FoldSeqAct where
  -- reduction with the companion operator corresponds to
  -- flattening. f must be semi-associative, with vecotor append (++)
  -- as the companion operator:
  --
  --   Forall b sh1 a1 sh2 a2.
--       f (f b sh1 a1) sh2 a2 = f b (sh1 ++ sh2) (a1 ++ a2).
  --
  -- It is common to ignore the shape vectors, yielding the usual
  -- semi-associativity law:
  --
  --   f b a _ = b + a,
  --
  -- for some (+) satisfying:
  --
  --   Forall b a1 a2. (b + a1) + a2 = b + (a1 ++ a2).
  --
  ExecFoldSeqFlatten :: (Arrays a, Shape sh, Elt e)
                     => !(ExecOpenAfun aenv (a -> Array (sh :. Int) e -> a))
                     -> !(ExecOpenAcc aenv a)
                     -> !(Idx senv (Array sh e))
                     -> ExecConsumer aenv senv a

  ExecStuple :: (Arrays a, IsAtuple a)
             => !(Atuple (ExecConsumer aenv senv) (TupleRepr a))
             -> ExecConsumer aenv senv a
