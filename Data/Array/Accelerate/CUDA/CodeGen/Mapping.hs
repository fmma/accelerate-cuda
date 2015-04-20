{-# LANGUAGE GADTs               #-}
{-# LANGUAGE ImpredicativeTypes  #-}
{-# LANGUAGE PatternGuards       #-}
{-# LANGUAGE QuasiQuotes         #-}
{-# LANGUAGE ScopedTypeVariables #-}
-- |
-- Module      : Data.Array.Accelerate.CUDA.CodeGen.Mapping
-- Copyright   : [2008..2014] Manuel M T Chakravarty, Gabriele Keller
--               [2009..2014] Trevor L. McDonell
-- License     : BSD3
--
-- Maintainer  : Trevor L. McDonell <tmcdonell@cse.unsw.edu.au>
-- Stability   : experimental
-- Portability : non-portable (GHC extensions)
--

module Data.Array.Accelerate.CUDA.CodeGen.Mapping (

  mkMap, mkZipWith,

) where

-- TODO remove
import Debug.Trace

import Language.C.Quote.CUDA
import Foreign.CUDA.Analysis.Device

import Data.Array.Accelerate.Array.Sugar                ( Array, Shape, Elt )
import Data.Array.Accelerate.CUDA.AST
import Data.Array.Accelerate.CUDA.CodeGen.Base


-- Apply the given unary function to each element of an array. Each thread
-- processes multiple elements, striding the array by the grid size.
--
-- map :: (Shape sh, Elt a, Elt b)
--     => (Exp a -> Exp b)
--     -> Acc (Array sh a)
--     -> Acc (Array sh b)
--
mkMap :: forall aenv senv sh a b. (Shape sh, Elt a, Elt b)
      => DeviceProperties
      -> Gamma aenv
      -> Gamma senv
      -> CUFun1 senv aenv (a -> b)
      -> CUDelayedAcc senv aenv sh a
      -> [CUTranslSkel senv aenv (Array sh b)]
mkMap dev aenv senv fun arr
  | CUFun1 dce f                 <- fun
  , CUDelayed _ _ (CUFun1 _ get) <- arr
  = return
  $ CUTranslSkel "map" [cunit|

    $esc:("#include <accelerate_cuda.h>")
    $edecls:texIn
    $edecls:sTexIn

    extern "C" __global__ void
    map
    (
        $params:argIn,
        $params:sArgIn,
        $params:argOut
    )
    {
        const int shapeSize     = $exp:(csize shOut);
        const int gridSize      = $exp:(gridSize dev);
              int ix;

        for ( ix =  $exp:(threadIdx dev)
            ; ix <  shapeSize
            ; ix += gridSize )
        {
            $items:(dce x       .=. get ix)
            $items:(setOut "ix" .=. f x)
        }
    }
  |]
  where
    (texIn, argIn)              = environment dev aenv
    (sTexIn, sArgIn)            = environmentS dev senv
    (argOut, shOut, setOut)     = writeArray "Out" (undefined :: Array sh b)
    (x, _, _)                   = locals "x" (undefined :: a)
    ix                          = [cvar "ix"]


-- Apply the binary unary function to each element of the two input
-- array. Each thread processes multiple elements, striding the array
-- by the grid size.
--
-- zipWith :: (Shape sh, Elt a, Elt b, Elt c)
--         => (Exp a -> Exp b -> Exp c)
--         -> Acc (Array sh a)
--         -> Acc (Array sh b)
--         -> Acc (Array sh c)
--
mkZipWith :: forall aenv senv sh a b c. (Shape sh, Elt a, Elt b, Elt c)
      => DeviceProperties
      -> Gamma aenv
      -> Gamma senv
      -> CUFun2 senv aenv (a -> b -> c)
      -> CUDelayedAcc senv aenv sh a
      -> CUDelayedAcc senv aenv sh c
      -> [CUTranslSkel senv aenv (Array sh c)]
mkZipWith dev aenv senv fun arra arrb
  | CUFun2 dcea dceb f           <- fun
  , CUDelayed _ _ (CUFun1 _ geta) <- arra
  , CUDelayed _ _ (CUFun1 _ getb) <- arrb
  = return
  $ CUTranslSkel "zipWith" [cunit|

    $esc:("#include <accelerate_cuda.h>")
    $edecls:texIn
    $edecls:sTexIn

    extern "C" __global__ void
    zipWith
    (
        $params:argIn,
        $params:sArgIn,
        $params:argOut
    )
    {
        const int shapeSize     = $exp:(csize shOut);
        const int gridSize      = $exp:(gridSize dev);
              int ix;

        for ( ix =  $exp:(threadIdx dev)
            ; ix <  shapeSize
            ; ix += gridSize )
        {
            $items:(dcea x       .=. geta ix)
            $items:(dceb y       .=. getb ix)
            $items:(setOut "ix" .=. f x y)
        }
    }
  |]
  where
    (texIn, argIn)              = environment dev aenv
    (sTexIn, sArgIn)            = environmentS dev senv
    (argOut, shOut, setOut)     = writeArray "Out" (undefined :: Array sh c)
    (x, _, _)                   = locals "x" (undefined :: a)
    (y, _, _)                   = locals "y" (undefined :: b)
    ix                          = [cvar "ix"]

