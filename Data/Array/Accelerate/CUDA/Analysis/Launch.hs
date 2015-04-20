{-# LANGUAGE GADTs                 #-}
{-# LANGUAGE RankNTypes            #-}
{-# LANGUAGE TemplateHaskell       #-}
{-# LANGUAGE ScopedTypeVariables   #-}
-- |
-- Module      : Data.Array.Accelerate.CUDA.Analysis.Launch
-- Copyright   : [2008..2014] Manuel M T Chakravarty, Gabriele Keller
--               [2009..2014] Trevor L. McDonell
-- License     : BSD3
--
-- Maintainer  : Trevor L. McDonell <tmcdonell@cse.unsw.edu.au>
-- Stability   : experimental
-- Portability : non-portable (GHC extensions)
--

module Data.Array.Accelerate.CUDA.Analysis.Launch (

  launchConfig, determineOccupancy

) where

-- friends
import Data.Array.Accelerate.AST
import Data.Array.Accelerate.Error
import Data.Array.Accelerate.Trafo
import Data.Array.Accelerate.Type
import Data.Array.Accelerate.Analysis.Type
import Data.Array.Accelerate.Analysis.Shape
import Data.Array.Accelerate.Array.Sugar

-- library
import qualified Foreign.CUDA.Analysis                  as CUDA
import qualified Foreign.CUDA.Driver                    as CUDA


-- |
-- Determine kernel launch parameters for the given array computation (as well
-- as compiled function module). This consists of the thread block size, number
-- of blocks, and dynamically allocated shared memory (bytes), respectively.
--
-- For most operations, this selects the minimum block size that gives maximum
-- occupancy, and the grid size limited to the maximum number of physically
-- resident blocks. Hence, kernels may need to process multiple elements per
-- thread. Scan operations select the largest block size of maximum occupancy.
--
launchConfig
    :: PreOpenArrayOp arr acc senv aenv a
    -> CUDA.DeviceProperties    -- the device being executed on
    -> CUDA.Occupancy           -- kernel occupancy information
    -> ( Int                    -- block size
       , Int -> Int             -- number of blocks for input problem size (grid)
       , Int )                  -- shared memory (bytes)
launchConfig op dev occ =
  let cta       = CUDA.activeThreads occ `div` CUDA.activeThreadBlocks occ
      maxGrid   = CUDA.multiProcessorCount dev * CUDA.activeThreadBlocks occ
      smem      = sharedMem dev op cta
  in
  (cta, \n -> maxGrid `min` gridSize dev op n cta, smem)


-- |
-- Determine maximal occupancy statistics for the given kernel / device
-- combination.
--
determineOccupancy
    :: PreOpenArrayOp arr acc senv aenv a
    -> CUDA.DeviceProperties
    -> CUDA.Fun                 -- corresponding __global__ entry function
    -> Int                      -- maximum number of threads per block
    -> IO CUDA.Occupancy
determineOccupancy op dev fn maxBlock = do
  registers     <- CUDA.requires fn CUDA.NumRegs
  static_smem   <- CUDA.requires fn CUDA.SharedSizeBytes        -- static memory only
  return . snd  $  blockSize dev op maxBlock registers (\threads -> static_smem + dynamic_smem threads)
  where
    dynamic_smem = sharedMem dev op


-- |
-- Determine an optimal thread block size for a given array computation. Fold
-- requires blocks with a power-of-two number of threads. Scans select the
-- largest size thread block possible, because if only one thread block is
-- needed we can calculate the scan in a single pass, rather than three.
--
blockSize
    :: CUDA.DeviceProperties
    -> PreOpenArrayOp arr acc senv aenv a
    -> Int                      -- maximum number of threads per block
    -> Int                      -- number of registers used
    -> (Int -> Int)             -- shared memory as a function of thread block size (bytes)
    -> (Int, CUDA.Occupancy)
blockSize dev op lim regs smem =
  CUDA.optimalBlockSizeBy dev (filter (<= lim) . strategy) (const regs) smem
  where
    strategy = case op of
      Fold{}    -> CUDA.incPow2
      Fold1{}   -> CUDA.incPow2
      Scanl{}   -> CUDA.incWarp
      Scanl'{}  -> CUDA.incWarp
      Scanl1{}  -> CUDA.incWarp
      Scanr{}   -> CUDA.incWarp
      Scanr'{}  -> CUDA.incWarp
      Scanr1{}  -> CUDA.incWarp
      _         -> CUDA.decWarp

-- |
-- Determine the number of blocks of the given size necessary to process the
-- given array expression. This should understand things like #elements per
-- thread for the various kernels.
--
-- The 'size' parameter is typically the number of elements in the array, except
-- for the following instances:
--
--  * foldSeg: the number of segments; require one warp per segment
--
--  * fold: for multidimensional reductions, this is the size of the shape tail
--          for 1D reductions this is the total number of elements
--
gridSize :: CUDA.DeviceProperties -> PreOpenArrayOp arr acc senv aenv a -> Int -> Int -> Int
gridSize p FoldSeg{}    sz cta = split (sz * CUDA.warpSize p) cta
gridSize p Fold1Seg{}   sz cta = split (sz * CUDA.warpSize p) cta
gridSize _ (Fold _ _ a) sz cta = if reifyDim a == 1 then split sz cta else max 1 sz
gridSize _ (Fold1 _ a)  sz cta = if reifyDim a == 1 then split sz cta else max 1 sz
gridSize _ _                 sz cta = split sz cta

split :: Int -> Int -> Int
split sz cta = (sz `between` eltsPerThread) `between` cta
  where
    between arr n   = 1 `max` ((n + arr - 1) `div` n)
    eltsPerThread   = 1


-- |
-- Analyse the given array expression, returning an estimate of dynamic shared
-- memory usage as a function of thread block size. This can be used by the
-- occupancy calculator to optimise kernel launch shape.
--
sharedMem :: CUDA.DeviceProperties -> PreOpenArrayOp arr acc senv aenv a -> Int -> Int
sharedMem _ Generate{}          _        = 0
sharedMem _ Transform{}         _        = 0
sharedMem _ Replicate{}         _        = 0
sharedMem _ Slice{}             _        = 0
sharedMem _ Map{}               _        = 0
sharedMem _ ZipWith{}           _        = 0
sharedMem _ Permute{}           _        = 0
sharedMem _ Backpermute{}       _        = 0
sharedMem _ Stencil{}           _        = 0
sharedMem _ Stencil2{}          _        = 0
sharedMem _ (Fold  _ _ a)       blockDim = sizeOf (reifyType a) * blockDim
sharedMem _ (Scanl _ _ a)       blockDim = sizeOf (reifyType a) * blockDim
sharedMem _ (Scanr _ _ a)       blockDim = sizeOf (reifyType a) * blockDim
sharedMem _ (Scanl' _ _ a)      blockDim = sizeOf (reifyType a) * blockDim
sharedMem _ (Scanr' _ _ a)      blockDim = sizeOf (reifyType a) * blockDim
sharedMem _ (Fold1 _ a)         blockDim = sizeOf (reifyType a) * blockDim
sharedMem _ (Scanl1 _ a)        blockDim = sizeOf (reifyType a) * blockDim
sharedMem _ (Scanr1 _ a)        blockDim = sizeOf (reifyType a) * blockDim
sharedMem p (FoldSeg _ _ a _)   blockDim =
  (blockDim `div` CUDA.warpSize p) * 8 + blockDim * sizeOf (reifyType a)  -- TLM: why 8? I can't remember...
sharedMem p (Fold1Seg _ a _) blockDim =
  (blockDim `div` CUDA.warpSize p) * 8 + blockDim * sizeOf (reifyType a)

