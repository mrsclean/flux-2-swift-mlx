// Flux2SingleBlock.swift - Single-Stream Transformer Block
// Copyright 2025 Vincent Gourbin

import Foundation
import MLX
import MLXNN

/// Single-Stream Transformer Block for Flux.2
///
/// Concatenates text and image hidden states, processes them together
/// with fused attention+FFN, then splits back. More efficient than
/// double-stream for later layers.
///
/// There are 48 such blocks in Flux.2.
public class Flux2SingleTransformerBlock: Module, @unchecked Sendable {
    let dim: Int
    let numHeads: Int
    let headDim: Int

    // Single LayerNorm for concatenated input
    let norm: LayerNorm

    // Fused parallel attention (QKV + MLP in one projection)
    let attn: Flux2ParallelSelfAttention

    /// Initialize Single-Stream Block
    /// - Parameters:
    ///   - dim: Model dimension (6144)
    ///   - numHeads: Number of attention heads (48)
    ///   - headDim: Dimension per head (128)
    ///   - mlpRatio: MLP expansion ratio
    public init(
        dim: Int,
        numHeads: Int,
        headDim: Int,
        mlpRatio: Float = 3.0
    ) {
        self.dim = dim
        self.numHeads = numHeads
        self.headDim = headDim

        self.norm = LayerNorm(dimensions: dim, eps: 1e-6, affine: false)
        self.attn = Flux2ParallelSelfAttention(
            dim: dim,
            numHeads: numHeads,
            headDim: headDim,
            mlpRatio: mlpRatio
        )
    }

    /// Forward pass
    /// - Parameters:
    ///   - hiddenStates: Image hidden states [B, S_img, dim]
    ///   - encoderHiddenStates: Text hidden states [B, S_txt, dim]
    ///   - temb: Timestep embedding [B, dim]
    ///   - rotaryEmb: Optional RoPE embeddings for combined sequence
    ///   - modParams: Modulation params (1 set for combined processing)
    /// - Returns: Updated image hidden states [B, S_img, dim]
    public func callAsFunction(
        hiddenStates: MLXArray,
        encoderHiddenStates: MLXArray?,
        temb: MLXArray,
        rotaryEmb: (cos: MLXArray, sin: MLXArray)? = nil,
        modParams: [ModulationParams]? = nil,
        refScaling: Flux2RefScalingContext? = nil,        // K4D LOCAL PATCH (Phase D)
        textTokenCount: Int? = nil                        // K4D LOCAL PATCH (Phase D)
    ) -> MLXArray {
        let residual = hiddenStates

        // If encoder_hidden_states is nil, hidden_states is assumed to already contain
        // concatenated text+image (diffusers pattern for single-stream blocks)
        let combined: MLXArray
        let textLen: Int
        if let encoderHS = encoderHiddenStates {
            // Old pattern: concatenate here
            combined = concatenated([encoderHS, hiddenStates], axis: 1)
            textLen = encoderHS.shape[1]
        } else {
            // New pattern: already concatenated. Caller MUST provide
            // textTokenCount if they want ref scaling on this block;
            // -1 means "skip ref scaling" (safe fallback).
            combined = hiddenStates
            textLen = textTokenCount ?? -1
        }

        // Normalize
        var normalized = norm(combined)

        // Apply modulation if provided
        if let mod = modParams, !mod.isEmpty {
            normalized = applyModulation(normalized, shift: mod[0].shift, scale: mod[0].scale)
        }

        // K4D LOCAL PATCH — per-block reference K/V strength (Phase D, 2026-05-20).
        // Single-stream `combined` is `[B, S_txt + S_img, dim]`. The ref-token
        // boundary in image-stream coordinates (refTokenStartInImage) maps to
        // combined-stream coordinates by adding textLen. Skipped when textLen
        // is unknown (caller didn't supply textTokenCount for the
        // already-concatenated path).
        if let ctx = refScaling,
           ctx.strength != 1.0,
           textLen >= 0 {
            let total = normalized.shape[1]
            let refStartInCombined = textLen + ctx.refTokenStartInImage
            if refStartInCombined < total {
                let headPart = normalized[0..., 0..<refStartInCombined, 0...]
                let refPart  = normalized[0..., refStartInCombined..<total, 0...] * MLXArray(ctx.strength)
                normalized = concatenated([headPart, refPart], axis: 1)
            }
        }
        // END K4D LOCAL PATCH

        // Parallel attention + FFN
        var output = attn(hiddenStates: normalized, rotaryEmb: rotaryEmb)

        // Apply gate if provided
        if let mod = modParams, !mod.isEmpty {
            output = applyGate(output, gate: mod[0].gate)
        }

        // Add residual and return full sequence
        // (the caller is responsible for splitting if needed)
        return residual + output
    }

    // MARK: - KV Cache Methods (for klein-9b-kv)

    /// Forward pass with KV extraction for single-stream blocks (step 0)
    public func callWithKVExtraction(
        hiddenStates: MLXArray,
        temb: MLXArray,
        rotaryEmb: (cos: MLXArray, sin: MLXArray)?,
        modParams: [ModulationParams]?,
        textLen: Int,
        referenceTokenCount: Int
    ) -> (MLXArray, LayerKVCacheEntry) {
        let residual = hiddenStates

        var normalized = norm(hiddenStates)

        if let mod = modParams, !mod.isEmpty {
            normalized = applyModulation(normalized, shift: mod[0].shift, scale: mod[0].scale)
        }

        let (attnOutput, cacheEntry) = attn.callWithKVExtraction(
            hiddenStates: normalized,
            rotaryEmb: rotaryEmb,
            textLen: textLen,
            referenceTokenCount: referenceTokenCount
        )

        var output = attnOutput
        if let mod = modParams, !mod.isEmpty {
            output = applyGate(output, gate: mod[0].gate)
        }

        return (residual + output, cacheEntry)
    }

    /// Forward pass with cached KV for single-stream blocks (steps 1+)
    public func callWithKVCached(
        hiddenStates: MLXArray,
        temb: MLXArray,
        rotaryEmb: (cos: MLXArray, sin: MLXArray)?,
        modParams: [ModulationParams]?,
        cachedKV: LayerKVCacheEntry,
        textLen: Int
    ) -> MLXArray {
        let residual = hiddenStates

        var normalized = norm(hiddenStates)

        if let mod = modParams, !mod.isEmpty {
            normalized = applyModulation(normalized, shift: mod[0].shift, scale: mod[0].scale)
        }

        let attnOutput = attn.callWithKVCached(
            hiddenStates: normalized,
            rotaryEmb: rotaryEmb,
            cachedKV: cachedKV,
            textLen: textLen
        )

        var output = attnOutput
        if let mod = modParams, !mod.isEmpty {
            output = applyGate(output, gate: mod[0].gate)
        }

        return residual + output
    }
}

/// Stack of Single-Stream Transformer Blocks
public class Flux2SingleTransformerBlocks: Module, @unchecked Sendable {
    let blocks: [Flux2SingleTransformerBlock]

    public init(
        numBlocks: Int,
        dim: Int,
        numHeads: Int,
        headDim: Int,
        mlpRatio: Float = 3.0
    ) {
        self.blocks = (0..<numBlocks).map { _ in
            Flux2SingleTransformerBlock(
                dim: dim,
                numHeads: numHeads,
                headDim: headDim,
                mlpRatio: mlpRatio
            )
        }
    }

    /// Forward pass through all single-stream blocks
    /// - Parameters:
    ///   - hiddenStates: Image hidden states [B, S_img, dim]
    ///   - encoderHiddenStates: Text hidden states [B, S_txt, dim] (used for concatenation)
    ///   - temb: Timestep embedding
    ///   - rotaryEmb: RoPE embeddings for combined sequence
    ///   - modulation: Modulation layer for timestep conditioning
    /// - Returns: Final image hidden states [B, S_img, dim]
    public func callAsFunction(
        hiddenStates: MLXArray,
        encoderHiddenStates: MLXArray,
        temb: MLXArray,
        rotaryEmb: (cos: MLXArray, sin: MLXArray)? = nil,
        modulation: Flux2Modulation? = nil
    ) -> MLXArray {
        var imgHS = hiddenStates

        for block in blocks {
            let modParams = modulation?(temb)

            imgHS = block(
                hiddenStates: imgHS,
                encoderHiddenStates: encoderHiddenStates,
                temb: temb,
                rotaryEmb: rotaryEmb,
                modParams: modParams
            )
        }

        return imgHS
    }
}

/// Variant with direct RoPE application per block
public class Flux2SingleTransformerBlockWithRoPE: Module, @unchecked Sendable {
    let dim: Int
    let numHeads: Int
    let headDim: Int

    let norm: LayerNorm
    let attn: Flux2ParallelSelfAttentionSplit  // Use split version for separate projections
    let rope: Flux2RoPE

    public init(
        dim: Int,
        numHeads: Int,
        headDim: Int,
        mlpRatio: Float = 3.0,
        ropeAxesDims: [Int] = [32, 32, 32, 32],
        ropeTheta: Float = 2000.0
    ) {
        self.dim = dim
        self.numHeads = numHeads
        self.headDim = headDim

        self.norm = LayerNorm(dimensions: dim, eps: 1e-6, affine: false)
        self.attn = Flux2ParallelSelfAttentionSplit(
            dim: dim,
            numHeads: numHeads,
            headDim: headDim,
            mlpRatio: mlpRatio
        )
        self.rope = Flux2RoPE(axesDims: ropeAxesDims, theta: ropeTheta)
    }

    public func callAsFunction(
        hiddenStates: MLXArray,
        encoderHiddenStates: MLXArray,
        temb: MLXArray,
        positionIds: MLXArray,
        modParams: [ModulationParams]? = nil
    ) -> MLXArray {
        let residual = hiddenStates
        let textSeqLen = encoderHiddenStates.shape[1]

        // Concatenate
        let combined = concatenated([encoderHiddenStates, hiddenStates], axis: 1)

        // Normalize
        var normalized = norm(combined)

        // Apply modulation
        if let mod = modParams, !mod.isEmpty {
            normalized = applyModulation(normalized, shift: mod[0].shift, scale: mod[0].scale)
        }

        // Generate RoPE embeddings from position IDs
        let ropeEmb = rope(positionIds)

        // Parallel attention + FFN with RoPE
        var output = attn(hiddenStates: normalized, rotaryEmb: ropeEmb)

        // Apply gate
        if let mod = modParams, !mod.isEmpty {
            output = applyGate(output, gate: mod[0].gate)
        }

        // Extract image portion and add residual
        let imgOutput = output[0..., textSeqLen..., 0...]
        return residual + imgOutput
    }
}
