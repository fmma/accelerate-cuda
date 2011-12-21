{-# LANGUAGE QuasiQuotes #-}
-- |
-- Module      : Data.Array.Accelerate.CUDA.CodeGen.IndexSpace
-- Copyright   : [2008..2011] Manuel M T Chakravarty, Gabriele Keller, Sean Lee, Trevor L. McDonell
-- License     : BSD3
--
-- Maintainer  : Manuel M T Chakravarty <chak@cse.unsw.edu.au>
-- Stability   : experimental
-- Portability : non-portable (GHC extensions)
--

module Data.Array.Accelerate.CUDA.CodeGen.IndexSpace (

  mkGenerate

) where

import Data.Loc
import Data.Symbol
import Language.C.Syntax
import Language.C.Quote.CUDA

import Data.Array.Accelerate.CUDA.CodeGen.Base


-- Construct a new array by applying a function to each index
--
-- generate :: (Shape ix, Elt e)
--          => Exp ix
--          -> (Exp ix -> Exp a)
--          -> Acc (Array ix a)
--
mkGenerate :: ([Type], Int) -> [Exp] -> CUTranslSkel
mkGenerate (tyOut, dimOut) fn =
  CUTranslSkel [cunit|
    $edecl:(dim "DimOut" dimOut)

    extern "C"
    __global__ void
    generate
    (
        $params:args,
        const typename DimOut shOut
    )
    {
              int idx;
        const int n        = size(shOut);
        const int gridSize = __umul24(blockDim.x, gridDim.x);

        for (idx = __umul24(blockDim.x, blockIdx.x) + threadIdx.x; idx < n; idx += gridSize)
        {
              $decls:shape
              $stms:(zipWith apply fn xs)
        }
    }
  |]
  where
    (xs, args)  = unzip (params "d_out" tyOut)
    apply f x   = [cstm| $exp:x [idx] = $exp:f; |]

    -- destruct shapes into separate components, since the code generator no
    -- longer treats tuples as structs
    --
    shape | dimOut == 1 = [[cdecl| const int x0_a0 = idx; |]]
          | otherwise   = sh : map (unsh . show) [0 .. dimOut-1]
          where
            sh      = [cdecl| const typename DimOut x0 = fromIndex(shOut, idx); |]
            unsh c  = [cdecl| const int $id:("x0_a" ++ c) = x0 . $id:('a':c); |]

