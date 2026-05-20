// Flux2Pipeline.swift - Main Pipeline for Flux.2 Image Generation
// Copyright 2025 Vincent Gourbin

import Foundation
import MLX
import MLXRandom
import MLXNN
import CoreGraphics
import ImageIO

#if canImport(AppKit)
import AppKit
#endif

/// Generation mode for Flux.2
// =========================================================================
// K4D LOCAL PATCH — strength-controlled img2img (2026-05-15)
// =========================================================================
// This file has additions for classical strength-controlled img2img
// (init-image + denoising-strength, SD-Auto1111-style). Marked with the
// banner comment "K4D LOCAL PATCH" everywhere it touches. Candidate for
// an upstream PR back to Vincent's repo — see commit notes.
//
// Extended 2026-05-15 with `.strengthImageToImageWithRefs` — combines
// VAE-encoded init (strength path) with concatenated Kontext refs
// (I2I path). Self-contained handler, no impact on the existing
// `.strengthImageToImage` (refs-empty) or `.imageToImage` paths.
//
// ROLLBACK:
//   1. Delete the .strengthImageToImage and .strengthImageToImageWithRefs
//      enum cases below.
//   2. Delete the `case .strengthImageToImage` branches added to the
//      `switch mode` in generate(mode:) and to the post-scheduler
//      blend block.
//   3. Delete the entire `case .strengthImageToImageWithRefs` handler
//      block in `switch mode` (self-contained, returns directly).
//   4. Revert the `strength: 1.0` line in the T2I scheduler call to its
//      hardcoded form (currently `strength: effectiveStrength`).
//   5. Delete the public `generateImageToImageWithStrength(...)` method
//      at the bottom of this file.
// All edits are additive; the existing T2I and I2I-kontext paths are
// untouched in their behavior.
// =========================================================================

public enum Flux2GenerationMode: Sendable {
    /// Text-to-Image generation
    case textToImage

    /// Image-to-Image generation with reference images
    /// - Parameter images: Reference images (1-3). Multiple images are concatenated along sequence dimension
    ///             with unique time-based position IDs for each, allowing the transformer to
    ///             attend to all reference images during generation.
    ///
    /// Note: Flux.2 I2I uses conditioning mode (not noise-injection like SD).
    /// Reference images provide visual context through transformer attention.
    /// The model always denoises from pure noise while attending to reference tokens.
    case imageToImage(images: [CGImage])

    /// K4D LOCAL PATCH — classical strength-controlled img2img.
    /// Start from a noised init latent (FlowMatch interpolation
    /// `x_t = (1-σ)·init + σ·noise`) instead of pure noise, then
    /// run only the tail of the denoise loop. `strength` ∈ (0, 1]:
    ///   - 1.0 → starting σ = 1.0 → init contribution vanishes,
    ///     equivalent to T2I behavior.
    ///   - 0.7 → starting σ ≈ 0.7 → init contributes ~30% at start.
    ///   - 0.5 → starting σ ≈ 0.5 → balanced init/noise mix.
    ///   - 0.2 → starting σ ≈ 0.2 → mostly init, light denoising.
    /// Klein is a 4-step distilled model; values ≥ 0.7 are
    /// recommended. Values below ~0.5 may degrade quality because
    /// the distilled checkpoint wasn't trained for partial
    /// trajectories.
    case strengthImageToImage(initImage: CGImage, strength: Float)

    /// K4D LOCAL PATCH — classical strength-controlled img2img with
    /// Kontext reference images concatenated as conditioning tokens.
    /// Combines `.strengthImageToImage` (init+strength latent setup)
    /// with `.imageToImage` (refs as transformer attention tokens) so
    /// a user can pin an init image AND attach 1–3 style refs in one
    /// render. Self-contained handler; does NOT run through the
    /// common denoise loop. Non-KV-cache implementation (parallels
    /// the standard I2I path, which also has both KV and non-KV
    /// variants).
    case strengthImageToImageWithRefs(initImage: CGImage, strength: Float, references: [CGImage])
}

/// Progress callback for generation (currentStep, totalSteps)
public typealias Flux2ProgressCallback = @Sendable (Int, Int) -> Void

/// Checkpoint callback for saving intermediate images (step, image)
public typealias Flux2CheckpointCallback = @Sendable (Int, CGImage) -> Void

// =====================================================================
// K4D LOCAL PATCH — reference channel mask (2026-05-17)
//
// Phase 1.1 of the Klein attention-control surface work. Klein's
// patchified reference latent is `[1, 128, H/16, W/16]`. Per
// capitan01R/ComfyUI-Flux2Klein-Enhancer's reverse-engineering:
// channels 0–63 carry structure/layout, channels 64–127 carry
// texture/detail. Zeroing one half lets a donor contribute only
// structure (low) or only texture (high) without changing the
// surrounding pipeline.
//
// Applied in `encodeReferenceImages` between BatchNorm and pack.
// Per-pipeline setting so K4D can flip it from CLI without
// changing every public generate method's signature.
//
// See `docs/klein-control-surface.md` in the K4D repo for the
// architecture facts and empirical-findings log.
// =====================================================================
public enum Flux2ReferenceChannelMask: String, Sendable {
    /// No masking — donor contributes via all 128 channels (default).
    case all
    /// Keep channels 0–63 only — donor contributes structure / layout.
    /// Channels 64–127 are zeroed before pack. Target use: silhouette /
    /// pose / shape donors where texture would muddy the binding.
    case low
    /// Keep channels 64–127 only — donor contributes texture / detail.
    /// Channels 0–63 are zeroed before pack. Target use: style /
    /// palette / texture donors where structure would imprint shape.
    case high
}
// END K4D LOCAL PATCH — reference channel mask

// =====================================================================
// K4D LOCAL PATCH — reference spatial fade (2026-05-17)
//
// Phase 1.3 (revised) of the Klein attention-control surface work.
// Per-position multiplier on the patchified ref latent, applied
// pre-pack alongside the channel mask. Lets a donor's spatial
// contribution be biased toward a region (center, edges, top, bottom,
// left, right) rather than uniform across the canvas.
//
// Implementation: a 2D float multiplier of shape `[H/16, W/16]`
// broadcast against the 128 channels. Each mode is a linear gradient
// from 1.0 to FADE_FLOOR (currently 0.2, hardcoded — keeps donor
// from going fully silent at the fade-out side).
//
// See `docs/klein-control-surface.md` for the lever catalog.
// =====================================================================
/// Parametric spatial fade (Phase 2.3 generalization, 2026-05-20).
///
/// Replaces the original 5-enum modes (none / center-out / edges-out /
/// top-down / left-right) with a continuous parameter space that
/// describes them all and arbitrary intermediate cases. The original
/// modes map cleanly to (shape + origin) combinations:
///
///   - `none` → `.off`
///   - `center-out` → `.radial`, origin=(0.5, 0.5)
///   - `top-down` → `.vertical`, origin=(_, 0.0)
///   - `left-right` → `.horizontal`, origin=(0.0, _)
///   - `edges-out` (inverse radial) is no longer expressible as a
///     single shape — was dropped in the rewrite. Users who want
///     "donor strong at edges" can place the origin off-canvas
///     (radius > distance to canvas center) for similar effect.
///
/// **Math:** at each patchified position `(y, x)` in `[H/16, W/16]`:
///   distance = (shape-dependent)
///   normDist = min(distance / radius, 1.0)
///   strength = 1.0 + (floor - 1.0) * normDist
/// → 1.0 at origin, linear ramp to `floor` at `distance == radius`,
///   clamped at `floor` beyond. All distances in canvas-normalized
///   units `[0, 1]` so the same parameters work across canvas
///   aspect ratios.
public struct Flux2ReferenceSpatialFade: Sendable {
    public enum Shape: String, Sendable {
        /// No fade (default, no-op).
        case off
        /// Euclidean distance from origin. Concentric falloff.
        case radial
        /// |x - originX| only. Fade along the x-axis from the origin column.
        case horizontal
        /// |y - originY| only. Fade along the y-axis from the origin row.
        case vertical
    }

    /// Which falloff geometry to use. `.off` skips the fade entirely.
    public let shape: Shape
    /// Origin x in canvas-normalized coordinates `[0.0, 1.0]`
    /// (0.0 = left edge, 1.0 = right edge). Donor's contribution is
    /// strongest at this column for `.radial` and `.horizontal`.
    public let originX: Float
    /// Origin y in canvas-normalized coordinates `[0.0, 1.0]`
    /// (0.0 = top edge, 1.0 = bottom edge). Donor's contribution is
    /// strongest at this row for `.radial` and `.vertical`.
    public let originY: Float
    /// Distance from origin (in canvas-normalized units) at which
    /// the donor strength reaches `floor`. Smaller = tighter falloff.
    /// `1.0` reaches `floor` at the far edge of a unit-square canvas.
    /// Clamped to `[0.01, 2.0]`.
    public let radius: Float
    /// Donor strength at maximum distance. `1.0` = uniform (no-op);
    /// `0.0` = donor fully zeroed at the far edge. Clamped to `[0, 1]`.
    public let floor: Float

    public init(
        shape: Shape = .off,
        originX: Float = 0.5,
        originY: Float = 0.5,
        radius: Float = 1.0,
        floor: Float = 0.2
    ) {
        self.shape = shape
        self.originX = max(0, min(1, originX))
        self.originY = max(0, min(1, originY))
        self.radius = max(0.01, min(2.0, radius))
        self.floor = max(0, min(1, floor))
    }

    /// No-op spatial fade. Engine code short-circuits on `shape == .off`.
    public static let off = Flux2ReferenceSpatialFade(shape: .off)
}
// END K4D LOCAL PATCH — reference spatial fade

// =====================================================================
// K4D LOCAL PATCH — per-block reference K/V strength (2026-05-20)
//
// Phase D of the Klein attention-control surface work. capitan01R's
// hook-trace established that Klein-on-ComfyUI ships `attn1_patch`
// that scales K and V at the ref-token slice inside every attention
// block. We replicate the same per-block effect by scaling the
// image-hidden-state `imgNorm` at ref-token positions BEFORE each
// block's attention call. Because Klein's toQ/toK/toV are bias-free
// linears, scaling input is mathematically equivalent to scaling K
// and V post-projection. Q at ref positions is also scaled, but ref
// outputs are discarded at extraction so the user-visible effect is
// the same as capitan01R's K/V-only patch.
//
// Threaded as an optional parameter through Flux2Transformer2DModel
// → Flux2TransformerBlock + Flux2SingleBlock → applied right before
// the attention call. Default nil (or strength=1.0) is a perfect
// no-op — no allocations, no MLX ops added to the graph.
//
// See `docs/klein-control-surface.md` for the full lever catalog +
// empirical findings log.
// =====================================================================
public struct Flux2RefScalingContext: Sendable {
    /// Index in the IMAGE stream where reference tokens begin.
    /// Tokens `[0..<refTokenStartInImage]` are main_img output;
    /// tokens `[refTokenStartInImage...]` are references. The
    /// double-stream block uses this offset directly (image-only
    /// stream). The single-stream block adds its text-prefix length
    /// internally to find the ref slice in the combined stream.
    public let refTokenStartInImage: Int
    /// Multiplicative scalar applied to `imgNorm` at ref positions.
    /// `1.0` = no-op. Useful range ~0.0..3.0. Klein at strength=0.0
    /// effectively ignores refs; strength>1.0 amplifies ref pull
    /// (often produces over-sharpening artifacts past ~1.5).
    public let strength: Float

    public init(refTokenStartInImage: Int, strength: Float) {
        self.refTokenStartInImage = refTokenStartInImage
        self.strength = strength
    }
}
// END K4D LOCAL PATCH — per-block reference K/V strength

/// Context passed to a denoising step hook.
///
/// `sigmaNext == 0` on the final iteration — chains relying on RePaint-style
/// blending (e.g. masked inpainting) use this to restore the unchanged region
/// to the clean original-image latent on the last step.
public struct Flux2StepContext: Sendable {
    /// Zero-based step index within the denoising loop.
    public let stepIdx: Int
    /// Total number of denoising steps that will run for this generation.
    public let totalSteps: Int
    /// Current sigma value (noise level at the start of this step).
    public let sigma: Float
    /// Next sigma value (noise level after this step). Equals `0` on the final step.
    public let sigmaNext: Float
    /// Target image height in pixels.
    public let height: Int
    /// Target image width in pixels.
    public let width: Int
    /// Whether this is the I2I conditioning path (true) or the T2I path (false).
    public let isI2I: Bool
}

/// Per-step hook called inside the denoising loop, after the scheduler step.
///
/// Lets a chain (e.g. masked inpainting) inspect or transform the packed latents
/// `[1, seq_len, 128]` between steps. Return the latents unchanged to act as a
/// pure observer, or return a new tensor to influence subsequent steps.
public typealias Flux2StepHook = @Sendable (Flux2StepContext, MLXArray) -> MLXArray

/// Result of image generation including the image and metadata
public struct Flux2GenerationResult: Sendable {
    /// The generated image
    public let image: CGImage

    /// The prompt that was actually used for generation
    /// If upsampling was enabled, this contains the enhanced prompt
    public let usedPrompt: String

    /// Whether the prompt was upsampled/enhanced
    public let wasUpsampled: Bool

    /// The original prompt before any enhancement
    public let originalPrompt: String

    public init(image: CGImage, usedPrompt: String, wasUpsampled: Bool, originalPrompt: String) {
        self.image = image
        self.usedPrompt = usedPrompt
        self.wasUpsampled = wasUpsampled
        self.originalPrompt = originalPrompt
    }
}

/// Flux.2 Image Generation Pipeline
///
/// Supports:
/// - Flux.2 Dev (32B) - Mistral text encoder
/// - Flux.2 Klein 4B - Qwen3-4B text encoder (Apache 2.0)
/// - Flux.2 Klein 9B - Qwen3-8B text encoder (Non-commercial)
///
/// Two-phase pipeline for memory efficiency:
/// 1. Text encoding (unloaded after use)
/// 2. Image generation with Transformer + VAE
public class Flux2Pipeline: @unchecked Sendable {
    /// Model variant (dev, klein-4b, klein-9b)
    public let model: Flux2Model

    /// Quantization configuration
    public let quantization: Flux2QuantizationConfig

    /// Memory optimization settings for transformer inference
    public var memoryOptimization: MemoryOptimizationConfig

    /// Text encoder (Mistral - for Dev)
    private var textEncoder: Flux2TextEncoder?

    /// Klein text encoder (Qwen3 - for Klein 4B/9B)
    private var kleinEncoder: KleinTextEncoder?

    /// Diffusion transformer
    private var transformer: Flux2Transformer2DModel?

    /// VAE variant (standard or small-decoder)
    public let vaeVariant: ModelRegistry.VAEVariant

    /// VAE decoder
    private var vae: AutoencoderKLFlux2?

    /// Scheduler
    private let scheduler: FlowMatchEulerScheduler

    /// Memory manager
    private let memoryManager = Flux2MemoryManager.shared

    /// Model downloader
    private var downloader: Flux2ModelDownloader?

    /// LoRA adapter manager
    private var loraManager: LoRAManager?

    /// Active LoRA scheduler overrides (from loaded LoRA config)
    private var loraSchedulerOverrides: SchedulerOverrides?

    /// Whether models are loaded
    public private(set) var isLoaded: Bool = false

    /// Memory profile for GPU cache management (auto = dynamic based on RAM)
    public var memoryProfile: MemoryConfig.CacheProfile = .auto

    /// Clear cache every N denoising steps (0 = disabled)
    public var clearCacheEveryNSteps: Int = 5

    // K4D LOCAL PATCH — reference channel mask (2026-05-17).
    // Applied in `encodeReferenceImages` between BatchNorm and pack.
    // Default `.all` is a no-op so existing callers see no behavior change.
    /// Channel mask applied to every reference's patchified latent
    /// before pack. `.all` = no mask (default). See
    /// `Flux2ReferenceChannelMask` for semantics.
    public var referenceChannelMask: Flux2ReferenceChannelMask = .all

    // K4D LOCAL PATCH — per-block reference K/V strength (2026-05-20).
    // Applied inside Flux2TransformerBlock + Flux2SingleBlock before
    // each attention call, scaling imgNorm at ref-token positions.
    // Default 1.0 is a perfect no-op (block skips the scaling op
    // entirely when strength == 1.0). Useful range ~0.0..2.0.
    /// Per-block multiplicative scalar applied to `imgNorm` at ref
    /// positions before each block's attention forward. `1.0` = no-op
    /// (default). Mathematically equivalent to scaling K and V at
    /// ref positions because Klein's toQ/toK/toV are bias-free
    /// linears. Replicates capitan01R's `attn1_patch` strength
    /// behavior on the K4D-native path. See
    /// `docs/klein-control-surface.md` for context.
    public var referenceStrength: Float = 1.0

    // K4D LOCAL PATCH — reference spatial fade (2026-05-17).
    // Applied in `encodeReferenceImages` after the channel mask,
    // before pack. Default `.none` is a no-op.
    /// Spatial-fade direction applied as a per-position multiplier
    /// on every reference's patchified latent before pack. `.none`
    /// Parametric spatial fade (Phase 2.3, 2026-05-20). `.off` is a
    /// perfect no-op (default). See `Flux2ReferenceSpatialFade`.
    public var referenceSpatialFade: Flux2ReferenceSpatialFade = .off

    /// Initialize pipeline
    /// - Parameters:
    ///   - model: Model variant to use (default: .dev)
    ///   - quantization: Quantization settings for each component
    ///   - hfToken: HuggingFace token for gated models
    /// Initialize the Flux.2 pipeline
    /// - Parameters:
    ///   - model: Model variant (dev, klein-4b, klein-9b)
    ///   - quantization: Quantization configuration
    ///   - memoryOptimization: Memory optimization settings (nil = auto-detect based on system RAM)
    ///   - hfToken: HuggingFace token for model downloads
    public init(
        model: Flux2Model = .dev,
        quantization: Flux2QuantizationConfig = .balanced,
        memoryOptimization: MemoryOptimizationConfig? = nil,
        vaeVariant: ModelRegistry.VAEVariant = .smallDecoder,
        hfToken: String? = nil
    ) {
        self.model = model
        self.quantization = quantization
        self.vaeVariant = vaeVariant
        self.memoryOptimization = memoryOptimization ?? MemoryOptimizationConfig.recommended(
            forRAMGB: Flux2MemoryManager.shared.physicalMemoryGB
        )
        self.scheduler = FlowMatchEulerScheduler()
        self.downloader = hfToken != nil ? Flux2ModelDownloader(hfToken: hfToken) : Flux2ModelDownloader()
    }

    // MARK: - Model Loading

    // MARK: - Internal accessors for chain helpers
    //
    // Exposed only within `Flux2Core` so the in-module
    // `Flux2Pipeline+ChainHelpers.swift` extension can reach the otherwise
    // private model instances and helpers. Out-of-module callers go through
    // the public API on that extension.

    internal func ensureVAELoaded() async throws { try await loadVAE() }
    internal var vaeForChains: AutoencoderKLFlux2? { vae }
    internal func preprocessImageForVAEPublic(
        _ image: CGImage,
        targetHeight: Int,
        targetWidth: Int
    ) -> MLXArray {
        preprocessImageForVAE(image, targetHeight: targetHeight, targetWidth: targetWidth)
    }

    /// Load all required models
    /// - Parameter progressCallback: Optional callback for download progress
    public func loadModels(progressCallback: Flux2DownloadProgressCallback? = nil) async throws {
        // Check memory before loading
        let memCheck = memoryManager.checkTextEncodingPhase(config: quantization)
        if !memCheck.isOk {
            Flux2Debug.log("Memory warning: \(memCheck.message)")
        }

        // Download models if needed
        if !hasRequiredModels {
            try await downloadRequiredModels(progress: progressCallback)
        }

        isLoaded = true
        Flux2Debug.log("Pipeline ready for generation")
    }

    /// Download required models
    private func downloadRequiredModels(progress: Flux2DownloadProgressCallback?) async throws {
        guard let downloader = downloader else { return }

        for component in missingModels {
            if case .textEncoder = component {
                // Text encoder is handled separately by Flux2TextEncoder
                continue
            }
            _ = try await downloader.download(component, progress: progress)
        }
    }

    /// Load text encoder for Phase 1
    /// Uses Mistral for Dev, Qwen3 for Klein models
    private func loadTextEncoder() async throws {
        // Check if already loaded
        switch model {
        case .dev:
            guard textEncoder == nil else { return }
        case .klein4B, .klein4BBase, .klein9B, .klein9BBase, .klein9BKV:
            guard kleinEncoder == nil else { return }
        }

        memoryManager.logMemoryState()
        Flux2Debug.log("Loading text encoder for \(model.displayName)...")

        // Map quantization
        let mistralQuant: MistralQuantization
        switch quantization.textEncoder {
        case .bf16:
            mistralQuant = .bf16
        case .mlx8bit:
            mistralQuant = .mlx8bit
        case .mlx6bit:
            mistralQuant = .mlx6bit
        case .mlx4bit:
            mistralQuant = .mlx4bit
        }

        switch model {
        case .dev:
            textEncoder = Flux2TextEncoder(quantization: mistralQuant)
            try await textEncoder!.load()

        case .klein4B, .klein4BBase:
            kleinEncoder = KleinTextEncoder(variant: .klein4B, quantization: mistralQuant)
            try await kleinEncoder!.load()

        case .klein9B, .klein9BBase, .klein9BKV:
            kleinEncoder = KleinTextEncoder(variant: .klein9B, quantization: mistralQuant)
            try await kleinEncoder!.load()
        }

        memoryManager.logMemoryState()
    }

    /// Unload text encoder to free memory for transformer
    @MainActor
    private func unloadTextEncoder() {
        Flux2Debug.log("Unloading text encoder...")

        switch model {
        case .dev:
            textEncoder?.unload()
            textEncoder = nil
        case .klein4B, .klein4BBase, .klein9B, .klein9BBase, .klein9BKV:
            kleinEncoder?.unload()
            kleinEncoder = nil
        }

        // Force synchronization to ensure all GPU operations complete
        // This helps release memory before loading the next model
        eval([])

        memoryManager.fullCleanup()

        // Additional sync after cache clear
        eval([])

        // Log memory state to verify release
        memoryManager.logMemoryState()

        // For large models (Dev), give the system a moment to reclaim memory
        // This prevents the text encoder and transformer from overlapping in memory
        if model == .dev {
            Flux2Debug.log("Waiting for memory reclamation (Dev model)...")
            Thread.sleep(forTimeInterval: 0.5)
            memoryManager.fullCleanup()
            eval([])
        }
    }

    /// Load transformer for Phase 2
    private func loadTransformer() async throws {
        guard transformer == nil else { return }

        MLX.Memory.peakMemory = 0
        memoryManager.logMemoryState()
        Flux2Debug.log("Loading transformer for \(model.displayName)...")

        // Get the appropriate transformer variant based on model type and quantization
        let variant = ModelRegistry.TransformerVariant.variant(for: model, quantization: quantization.transformer)

        // Find model path
        guard let modelPath = Flux2ModelDownloader.findModelPath(for: .transformer(variant)) else {
            let downloadCmd: String
            switch model {
            case .dev:
                downloadCmd = "flux2 download --transformer \(quantization.transformer.rawValue)"
            case .klein4B, .klein4BBase:
                downloadCmd = "flux2 download --model klein-4b"
            case .klein9B, .klein9BBase:
                downloadCmd = "flux2 download --model klein-9b"
            case .klein9BKV:
                downloadCmd = "flux2 download --model klein-9b-kv"
            }
            throw Flux2Error.modelNotLoaded("\(model.displayName) transformer weights not found. Run: \(downloadCmd)")
        }

        // Create model with appropriate config and memory optimization
        transformer = Flux2Transformer2DModel(
            config: model.transformerConfig,
            memoryOptimization: memoryOptimization
        )
        Flux2Debug.log("Memory optimization: \(memoryOptimization)")

        // Load weights with explicit memory management
        // For large models (Dev), this can temporarily use 2x memory during mapping
        Flux2Debug.log("Loading transformer weights from disk...")
        var weights = try Flux2WeightLoader.loadWeights(from: modelPath)

        Flux2Debug.log("Applying weights to model...")
        try Flux2WeightLoader.applyTransformerWeights(&weights, to: transformer!)

        // Explicitly release the raw weights dictionary to free memory
        // This is important for Dev model where weights can be ~32GB
        weights.removeAll()
        eval([])  // Sync to ensure weights are released
        memoryManager.fullCleanup()
        Flux2Debug.log("Raw weights released from memory")

        // Quantize transformer to native MLX QuantizedLinear if requested
        // This handles both:
        // 1. Pre-quantized weights (quanto→float16→MLX qint8): negligible precision loss
        // 2. On-the-fly quantization from bf16 (e.g. Klein 9B qint8, any model int4)
        // The quantization uses MLX's native QuantizedLinear format which:
        // - Reduces memory usage proportionally to bit width (8-bit: ~50%, 4-bit: ~75%)
        // - Uses optimized quantizedMM() for faster inference on Apple Silicon
        // - Enables efficient dequant→merge→requant for LoRA weight merging
        if quantization.transformer != .bf16 {
            let bits = quantization.transformer.bits
            let groupSize = quantization.transformer.groupSize
            Flux2Debug.log("Quantizing transformer on-the-fly to \(bits)-bit (groupSize=\(groupSize))...")
            memoryManager.logMemoryState()
            quantize(model: transformer!, groupSize: groupSize, bits: bits)
            eval(transformer!.parameters())
            memoryManager.fullCleanup()
            memoryManager.logMemoryState()
            Flux2Debug.log("Transformer quantized to QuantizedLinear (\(bits)-bit)")
        }

        // Merge LoRA weights if any are loaded
        if let loraManager = loraManager, loraManager.count > 0 {
            MLX.Memory.peakMemory = 0
            Flux2Debug.log("[LoRA] Before merge:")
            memoryManager.logMemoryState()
            Flux2WeightLoader.mergeLoRAWeights(from: loraManager, into: transformer!)
            // Free LoRA matrices from memory after fusion (they're now baked into base weights)
            loraManager.clearWeightsAfterFusion()
            memoryManager.fullCleanup()
            Flux2Debug.log("[LoRA] After merge:")
            memoryManager.logMemoryState()
        }

        // Ensure weights are evaluated
        eval(transformer!.parameters())

        memoryManager.logMemoryState()
        Flux2Debug.log("Transformer loaded successfully")
    }

    // MARK: - LoRA Support

    /// Load a LoRA adapter
    /// - Parameter config: LoRA configuration
    /// - Returns: Information about the loaded LoRA
    @discardableResult
    public func loadLoRA(_ config: LoRAConfig) throws -> LoRAInfo {
        if loraManager == nil {
            loraManager = LoRAManager()
        }

        let info = try loraManager!.loadLoRA(config)

        // Store scheduler overrides if present
        if let overrides = config.schedulerOverrides, overrides.hasOverrides {
            loraSchedulerOverrides = overrides
            Flux2Debug.log("[LoRA] Scheduler overrides detected:")
            if let sigmas = overrides.customSigmas {
                Flux2Debug.log("  - Custom sigmas: \(sigmas.count) values")
            }
            if let steps = overrides.numSteps {
                Flux2Debug.log("  - Recommended steps: \(steps)")
            }
            if let guidance = overrides.guidance {
                Flux2Debug.log("  - Recommended guidance: \(guidance)")
            }
        }

        // Validate compatibility
        if info.targetModel != .unknown {
            let expectedModel: LoRAInfo.TargetModel
            switch model {
            case .dev: expectedModel = .dev
            case .klein4B, .klein4BBase: expectedModel = .klein4B
            case .klein9B, .klein9BBase, .klein9BKV: expectedModel = .klein9B
            }

            if info.targetModel != expectedModel {
                Flux2Debug.log("[LoRA] Warning: LoRA was trained for \(info.targetModel.rawValue), but using \(model.rawValue)")
            }
        }

        // If transformer is already loaded, merge LoRA weights immediately
        if let transformer = transformer {
            Flux2WeightLoader.mergeLoRAWeights(from: loraManager!, into: transformer)
            // Free LoRA matrices from memory after fusion (they're now baked into base weights)
            loraManager!.clearWeightsAfterFusion()
        }

        return info
    }

    /// Unload a LoRA by name
    public func unloadLoRA(name: String) {
        loraManager?.unloadLoRA(name: name)
        // Clear overrides if no more LoRAs are loaded
        if loraManager?.count == 0 {
            loraSchedulerOverrides = nil
        }
    }

    /// Unload all LoRAs
    public func unloadAllLoRAs() {
        loraManager?.unloadAll()
        loraSchedulerOverrides = nil
    }

    /// Whether any LoRAs are loaded
    public var hasLoRA: Bool {
        loraManager?.count ?? 0 > 0
    }

    /// Get active LoRA scheduler overrides (if any)
    /// The CLI can use this to apply recommended settings
    public var activeSchedulerOverrides: SchedulerOverrides? {
        loraSchedulerOverrides
    }

    /// Load VAE for Phase 2
    private func loadVAE() async throws {
        guard vae == nil else { return }

        Flux2Debug.log("Loading VAE (\(vaeVariant.displayName))...")

        guard let modelPath = Flux2ModelDownloader.findModelPath(for: .vae(vaeVariant)) else {
            throw Flux2Error.modelNotLoaded("VAE weights not found for variant: \(vaeVariant.rawValue)")
        }

        // VAE files may be in 'vae' subdirectory (standard variant from Klein 4B repo)
        let vaePath = modelPath.appendingPathComponent("vae")
        let weightsPath = FileManager.default.fileExists(atPath: vaePath.path) ? vaePath : modelPath

        // Load config from JSON if available, otherwise use variant preset
        let configURL = weightsPath.appendingPathComponent("config.json")
        let vaeConfig: VAEConfig
        if FileManager.default.fileExists(atPath: configURL.path) {
            vaeConfig = try VAEConfig.load(from: configURL)
            Flux2Debug.log("VAE config loaded from config.json (decoder channels: \(vaeConfig.effectiveDecoderChannels))")
        } else {
            vaeConfig = vaeVariant.vaeConfig
            Flux2Debug.log("VAE using preset config for \(vaeVariant.rawValue)")
        }

        // Create model with appropriate config
        vae = AutoencoderKLFlux2(config: vaeConfig)

        // Load weights — prefer diffusion_pytorch_model.safetensors (standard diffusers file)
        // to avoid conflicts when directory contains multiple safetensors files
        let standardWeightsFile = weightsPath.appendingPathComponent("diffusion_pytorch_model.safetensors")
        let weights: [String: MLXArray]
        if FileManager.default.fileExists(atPath: standardWeightsFile.path) {
            Flux2Debug.log("Loading VAE weights from diffusion_pytorch_model.safetensors")
            weights = try Flux2WeightLoader.loadWeights(from: standardWeightsFile)
        } else {
            weights = try Flux2WeightLoader.loadWeights(from: weightsPath)
        }
        try Flux2WeightLoader.applyVAEWeights(weights, to: vae!)

        // Ensure weights are evaluated
        eval(vae!.parameters())

        Flux2Debug.log("VAE loaded successfully (\(vaeVariant.displayName), decoder: \(vaeConfig.effectiveDecoderChannels))")
    }

    /// Unload transformer to free memory
    private func unloadTransformer() {
        transformer = nil
        memoryManager.clearCache()
    }

    // MARK: - Generation API

    /// Generate image from text prompt
    /// - Parameters:
    ///   - prompt: Text description of the image
    ///   - height: Image height (default 1024)
    ///   - width: Image width (default 1024)
    ///   - steps: Number of denoising steps (default 50)
    ///   - guidance: Guidance scale (default 4.0)
    ///   - seed: Optional random seed
    ///   - upsamplePrompt: Enhance prompt with visual details before encoding (default false)
    ///   - checkpointInterval: Save intermediate image every N steps (nil = disabled)
    ///   - onProgress: Optional progress callback
    ///   - onCheckpoint: Optional callback when checkpoint image is generated
    /// - Returns: Generated image
    public func generateTextToImage(
        prompt: String,
        interpretImagePaths: [String]? = nil,
        height: Int = 1024,
        width: Int = 1024,
        steps: Int = 50,
        guidance: Float = 4.0,
        seed: UInt64? = nil,
        upsamplePrompt: Bool = false,
        precomputedEmbeddings: MLXArray? = nil,
        checkpointInterval: Int? = nil,
        onProgress: Flux2ProgressCallback? = nil,
        onCheckpoint: Flux2CheckpointCallback? = nil
    ) async throws -> CGImage {
        try await generate(
            mode: .textToImage,
            prompt: prompt,
            interpretImagePaths: interpretImagePaths,
            height: height,
            width: width,
            steps: steps,
            guidance: guidance,
            seed: seed,
            upsamplePrompt: upsamplePrompt,
            precomputedEmbeddings: precomputedEmbeddings,
            checkpointInterval: checkpointInterval,
            onProgress: onProgress,
            onCheckpoint: onCheckpoint
        )
    }

    /// Generate image with reference images
    ///
    /// Flux.2 uses conditioning mode: reference images provide visual context through
    /// transformer attention. The model always denoises from pure noise.
    ///
    /// - Parameters:
    ///   - prompt: Text description
    ///   - images: 1-3 reference images. Multiple images are concatenated along sequence dimension
    ///             with unique time-based position IDs, allowing the transformer to attend to all references.
    ///   - interpretImagePaths: File paths to images to analyze with VLM and inject description into prompt (not used as visual reference)
    ///   - height: Optional height (inferred from first image if nil)
    ///   - width: Optional width (inferred from first image if nil)
    ///   - steps: Number of denoising steps
    ///   - guidance: Guidance scale
    ///   - seed: Optional random seed
    ///   - upsamplePrompt: Enhance prompt with visual details before encoding (default false)
    ///   - checkpointInterval: Save intermediate image every N steps (nil = disabled)
    ///   - onProgress: Optional progress callback
    ///   - onCheckpoint: Optional callback when checkpoint image is generated
    /// - Returns: Generated image
    public func generateImageToImage(
        prompt: String,
        images: [CGImage],
        interpretImagePaths: [String]? = nil,
        height: Int? = nil,
        width: Int? = nil,
        steps: Int = 50,
        guidance: Float = 4.0,
        seed: UInt64? = nil,
        upsamplePrompt: Bool = false,
        checkpointInterval: Int? = nil,
        onProgress: Flux2ProgressCallback? = nil,
        onCheckpoint: Flux2CheckpointCallback? = nil
    ) async throws -> CGImage {
        guard !images.isEmpty && images.count <= 3 else {
            throw Flux2Error.invalidConfiguration("Provide 1-3 reference images")
        }

        // Infer dimensions from first image if not provided
        let targetHeight = height ?? images[0].height
        let targetWidth = width ?? images[0].width

        return try await generate(
            mode: .imageToImage(images: images),
            prompt: prompt,
            interpretImagePaths: interpretImagePaths,
            height: targetHeight,
            width: targetWidth,
            steps: steps,
            guidance: guidance,
            seed: seed,
            upsamplePrompt: upsamplePrompt,
            checkpointInterval: checkpointInterval,
            onProgress: onProgress,
            onCheckpoint: onCheckpoint
        )
    }

    /// Generate image with reference images from raw image data (PNG/JPEG)
    /// - Note: Uses CGImageSource for pixel-exact decoding, avoiding NSImage roundtrip
    ///         which can introduce subpixel shifts via AppKit re-rendering.
    public func generateImageToImage(
        prompt: String,
        imageData: [Data],
        interpretImagePaths: [String]? = nil,
        height: Int? = nil,
        width: Int? = nil,
        steps: Int = 50,
        guidance: Float = 4.0,
        seed: UInt64? = nil,
        upsamplePrompt: Bool = false,
        checkpointInterval: Int? = nil,
        onProgress: Flux2ProgressCallback? = nil,
        onCheckpoint: Flux2CheckpointCallback? = nil
    ) async throws -> CGImage {
        let images = try imageData.enumerated().map { index, data in
            guard let cgImage = Self.cgImage(from: data) else {
                throw Flux2Error.invalidConfiguration("Failed to decode image data at index \(index)")
            }
            return cgImage
        }
        return try await generateImageToImage(
            prompt: prompt,
            images: images,
            interpretImagePaths: interpretImagePaths,
            height: height,
            width: width,
            steps: steps,
            guidance: guidance,
            seed: seed,
            upsamplePrompt: upsamplePrompt,
            checkpointInterval: checkpointInterval,
            onProgress: onProgress,
            onCheckpoint: onCheckpoint
        )
    }

    // MARK: - Generation with Result (includes used prompt)

    /// Generate an image from text with full result including the used prompt
    /// - Returns: Flux2GenerationResult containing image and prompt metadata
    public func generateTextToImageWithResult(
        prompt: String,
        interpretImagePaths: [String]? = nil,
        height: Int = 1024,
        width: Int = 1024,
        steps: Int = 50,
        guidance: Float = 4.0,
        seed: UInt64? = nil,
        upsamplePrompt: Bool = false,
        checkpointInterval: Int? = nil,
        onProgress: Flux2ProgressCallback? = nil,
        onCheckpoint: Flux2CheckpointCallback? = nil
    ) async throws -> Flux2GenerationResult {
        try await generateWithResult(
            mode: .textToImage,
            prompt: prompt,
            interpretImagePaths: interpretImagePaths,
            height: height,
            width: width,
            steps: steps,
            guidance: guidance,
            seed: seed,
            upsamplePrompt: upsamplePrompt,
            checkpointInterval: checkpointInterval,
            onProgress: onProgress,
            onCheckpoint: onCheckpoint
        )
    }

    /// Generate an image from reference images with full result including the used prompt
    /// - Returns: Flux2GenerationResult containing image and prompt metadata
    public func generateImageToImageWithResult(
        prompt: String,
        images: [CGImage],
        interpretImagePaths: [String]? = nil,
        height: Int? = nil,
        width: Int? = nil,
        steps: Int = 50,
        guidance: Float = 4.0,
        seed: UInt64? = nil,
        upsamplePrompt: Bool = false,
        checkpointInterval: Int? = nil,
        onProgress: Flux2ProgressCallback? = nil,
        onCheckpoint: Flux2CheckpointCallback? = nil
    ) async throws -> Flux2GenerationResult {
        guard !images.isEmpty && images.count <= 3 else {
            throw Flux2Error.invalidConfiguration("Provide 1-3 reference images")
        }

        // Infer dimensions from first image if not provided
        let targetHeight = height ?? images[0].height
        let targetWidth = width ?? images[0].width

        return try await generateWithResult(
            mode: .imageToImage(images: images),
            prompt: prompt,
            interpretImagePaths: interpretImagePaths,
            height: targetHeight,
            width: targetWidth,
            steps: steps,
            guidance: guidance,
            seed: seed,
            upsamplePrompt: upsamplePrompt,
            checkpointInterval: checkpointInterval,
            onProgress: onProgress,
            onCheckpoint: onCheckpoint
        )
    }

    /// Generate image with reference images from raw image data with full result
    /// - Note: Uses CGImageSource for pixel-exact decoding, avoiding NSImage roundtrip.
    /// - Returns: Flux2GenerationResult containing image and prompt metadata
    public func generateImageToImageWithResult(
        prompt: String,
        imageData: [Data],
        interpretImagePaths: [String]? = nil,
        height: Int? = nil,
        width: Int? = nil,
        steps: Int = 50,
        guidance: Float = 4.0,
        seed: UInt64? = nil,
        upsamplePrompt: Bool = false,
        checkpointInterval: Int? = nil,
        onProgress: Flux2ProgressCallback? = nil,
        onCheckpoint: Flux2CheckpointCallback? = nil
    ) async throws -> Flux2GenerationResult {
        let images = try imageData.enumerated().map { index, data in
            guard let cgImage = Self.cgImage(from: data) else {
                throw Flux2Error.invalidConfiguration("Failed to decode image data at index \(index)")
            }
            return cgImage
        }
        return try await generateImageToImageWithResult(
            prompt: prompt,
            images: images,
            interpretImagePaths: interpretImagePaths,
            height: height,
            width: width,
            steps: steps,
            guidance: guidance,
            seed: seed,
            upsamplePrompt: upsamplePrompt,
            checkpointInterval: checkpointInterval,
            onProgress: onProgress,
            onCheckpoint: onCheckpoint
        )
    }

    /// Unified generation method (backward compatible - returns just the image)
    public func generate(
        mode: Flux2GenerationMode,
        prompt: String,
        interpretImagePaths: [String]? = nil,
        height: Int,
        width: Int,
        steps: Int,
        guidance: Float,
        seed: UInt64?,
        upsamplePrompt: Bool,
        precomputedEmbeddings: MLXArray? = nil,
        checkpointInterval: Int?,
        onProgress: Flux2ProgressCallback?,
        onCheckpoint: Flux2CheckpointCallback?,
        onStep: Flux2StepHook? = nil
    ) async throws -> CGImage {
        let result = try await generateWithResult(
            mode: mode,
            prompt: prompt,
            interpretImagePaths: interpretImagePaths,
            height: height,
            width: width,
            steps: steps,
            guidance: guidance,
            seed: seed,
            upsamplePrompt: upsamplePrompt,
            precomputedEmbeddings: precomputedEmbeddings,
            checkpointInterval: checkpointInterval,
            onProgress: onProgress,
            onCheckpoint: onCheckpoint,
            onStep: onStep
        )
        return result.image
    }

    /// Unified generation method with full result including used prompt
    public func generateWithResult(
        mode: Flux2GenerationMode,
        prompt: String,
        interpretImagePaths: [String]? = nil,
        height: Int,
        width: Int,
        steps: Int,
        guidance: Float,
        seed: UInt64?,
        upsamplePrompt: Bool,
        precomputedEmbeddings: MLXArray? = nil,
        checkpointInterval: Int?,
        onProgress: Flux2ProgressCallback?,
        onCheckpoint: Flux2CheckpointCallback?,
        onStep: Flux2StepHook? = nil
    ) async throws -> Flux2GenerationResult {
        // Validate dimensions
        let (validHeight, validWidth) = LatentUtils.validateDimensions(
            height: height,
            width: width
        )

        // Check image size feasibility
        let sizeCheck = memoryManager.checkImageSize(width: validWidth, height: validHeight)
        if case .insufficientMemory = sizeCheck {
            throw Flux2Error.insufficientMemory(required: 100, available: memoryManager.estimatedAvailableMemoryGB)
        }

        // Set random seed
        if let seed = seed {
            MLXRandom.seed(seed)
        }

        Flux2Debug.log("Starting generation: \(validWidth)x\(validHeight), \(steps) steps, guidance=\(guidance)")

        // Start profiling
        let profiler = Flux2Profiler.shared

        // === PHASE 1: Text Encoding ===
        // Note: Progress will be reported once denoising loop starts with accurate step count

        let textEmbeddings: MLXArray
        var finalUsedPrompt: String = prompt
        var wasPromptUpsampled: Bool = false

        // Classical CFG (klein base only): cache an empty-prompt embedding to
        // run the transformer twice per step. Skipped at guidance ≤ 1.
        let useClassicalCFG = model.usesClassicalCFG && guidance > 1.0
        var negativeTextEmbeddings: MLXArray? = nil

        // MEMORY OPTIMIZATION: Compute phase limits for all phases
        let phaseLimits = MemoryConfig.PhaseLimits.forModel(model, profile: memoryProfile)

        if let precomputed = precomputedEmbeddings {
            // Skip text encoding entirely — use pre-computed embeddings
            textEmbeddings = precomputed
            Flux2Debug.log("=== PHASE 1: SKIPPED (using pre-computed embeddings) ===")
            Flux2Debug.log("Pre-computed embeddings shape: \(precomputed.shape)")
        } else {
            Flux2Debug.log("=== PHASE 1: Text Encoding ===")

            MemoryConfig.applyCacheLimit(bytes: phaseLimits.textEncoding)

            profiler.start("1. Load Text Encoder")
            try await loadTextEncoder()
            profiler.end("1. Load Text Encoder")

            // === INTERPRET IMAGES: VLM semantic analysis ===
            // If interpretImagePaths are provided, describe them with VLM and inject into prompt
            // Note: VLM interpretation only available for Dev model (Mistral)
            var enrichedPrompt = prompt
            if let interpretPaths = interpretImagePaths, !interpretPaths.isEmpty {
                switch model {
                case .dev:
                    Flux2Debug.log("Interpreting \(interpretPaths.count) image(s) with VLM for prompt injection...")
                    profiler.start("1b. VLM Interpretation")

                    let descriptions = try await textEncoder!.describeImagePathsForPrompt(interpretPaths, context: prompt)

                    if !descriptions.isEmpty {
                        // Build enriched prompt with image descriptions
                        let imageContext = descriptions.enumerated().map { (idx, desc) in
                            "Interpret image \(idx + 1): \(desc)"
                        }.joined(separator: "\n")

                        enrichedPrompt = """
                        \(imageContext)

                        User request: \(prompt)
                        """

                        Flux2Debug.log("Prompt enriched with \(descriptions.count) VLM description(s)")
                        Flux2Debug.info("[VLM-Interpret] Enriched prompt:\n\(enrichedPrompt)")
                    }

                    profiler.end("1b. VLM Interpretation")

                case .klein4B, .klein4BBase, .klein9B, .klein9BBase, .klein9BKV:
                    // Klein + --interpret: load Mistral VLM temporarily to analyze images
                    Flux2Debug.log("Klein with --interpret: loading Mistral VLM temporarily to analyze images...")
                    profiler.start("1b. VLM Interpretation")

                    // Step 1: Unload Qwen3 to free memory for Mistral
                    Flux2Debug.log("Unloading Qwen3 to make room for Mistral VLM...")
                    await MainActor.run { kleinEncoder?.unload() }
                    memoryManager.fullCleanup()

                    // Step 2: Load Mistral VLM
                    Flux2Debug.log("Loading Mistral VLM for image interpretation...")
                    let tempMistralForInterpret = Flux2TextEncoder(quantization: quantization.textEncoder)
                    try await tempMistralForInterpret.load()

                    // Step 3: VLM interpretation (same logic as Dev)
                    let descriptions = try await tempMistralForInterpret.describeImagePathsForPrompt(interpretPaths, context: prompt)

                    if !descriptions.isEmpty {
                        let imageContext = descriptions.enumerated().map { (idx, desc) in
                            "Interpret image \(idx + 1): \(desc)"
                        }.joined(separator: "\n")

                        enrichedPrompt = """
                        \(imageContext)

                        User request: \(prompt)
                        """

                        Flux2Debug.log("Prompt enriched with \(descriptions.count) VLM description(s)")
                        Flux2Debug.info("[VLM-Interpret] Enriched prompt:\n\(enrichedPrompt)")
                    }

                    // Step 4: Unload Mistral
                    Flux2Debug.log("Unloading Mistral VLM...")
                    await MainActor.run { tempMistralForInterpret.unload() }
                    memoryManager.fullCleanup()

                    // Step 5: Reload Qwen3 for text encoding
                    Flux2Debug.log("Reloading Qwen3 for Klein text encoding...")
                    try await kleinEncoder!.load()

                    profiler.end("1b. VLM Interpretation")
                }
            }

            profiler.start("2. Text Encoding")
            // Use vision-based upsampling for I2I if enabled (Dev only)
            // Track the final prompt used for generation

            switch model {
            case .dev:
                if upsamplePrompt, case .imageToImage(let images) = mode {
                    // Use VLM to analyze reference images and enhance prompt
                    Flux2Debug.log("Using vision-based prompt upsampling for I2I with \(images.count) image(s)")
                    let enhancedPrompt = try await textEncoder!.upsamplePromptWithImages(enrichedPrompt, images: images)
                    finalUsedPrompt = enhancedPrompt
                    wasPromptUpsampled = true
                    textEmbeddings = try textEncoder!.encode(enhancedPrompt, upsample: false)
                } else {
                    let (embeddings, usedPrompt) = try textEncoder!.encodeWithPrompt(enrichedPrompt, upsample: upsamplePrompt)
                    textEmbeddings = embeddings
                    finalUsedPrompt = usedPrompt
                    wasPromptUpsampled = upsamplePrompt && (usedPrompt != enrichedPrompt)
                }

            case .klein4B, .klein4BBase, .klein9B, .klein9BBase, .klein9BKV:
                // Klein I2I with upsampling: load Mistral VLM temporarily to see reference images
                // This matches the official flux2 implementation which loads Mistral for Klein I2I upsampling
                if upsamplePrompt, case .imageToImage(let images) = mode {
                    Flux2Debug.log("Klein I2I with upsampling: using Mistral VLM to analyze reference images...")

                    // Step 1: Unload Qwen3 (already loaded by loadTextEncoder) to free memory for Mistral
                    Flux2Debug.log("Unloading Qwen3 to make room for Mistral VLM...")
                    await MainActor.run { kleinEncoder?.unload() }
                    memoryManager.fullCleanup()

                    // Step 2: Load Mistral VLM for vision-aware upsampling
                    Flux2Debug.log("Loading Mistral VLM for image analysis...")
                    let tempMistralEncoder = Flux2TextEncoder(quantization: quantization.textEncoder)
                    try await tempMistralEncoder.load()

                    // Step 3: Upsample prompt with images using Mistral VLM
                    Flux2Debug.log("Upsampling prompt with \(images.count) reference image(s)...")
                    let enhancedPrompt = try await tempMistralEncoder.upsamplePromptWithImages(enrichedPrompt, images: images)
                    finalUsedPrompt = enhancedPrompt
                    wasPromptUpsampled = true

                    // Step 4: Unload Mistral to free memory
                    Flux2Debug.log("Unloading Mistral VLM...")
                    await MainActor.run { tempMistralEncoder.unload() }
                    memoryManager.fullCleanup()

                    // Step 5: Reload Qwen3 for text encoding
                    Flux2Debug.log("Reloading Qwen3 for Klein text encoding...")
                    try await kleinEncoder!.load()

                    // Step 6: Encode with Qwen3 (already upsampled, so upsample=false)
                    textEmbeddings = try kleinEncoder!.encode(enhancedPrompt, upsample: false)
                } else {
                    // Standard Klein encoding (text-only upsampling if enabled)
                    let (embeddings, usedPrompt) = try kleinEncoder!.encodeWithPrompt(enrichedPrompt, upsample: upsamplePrompt)
                    textEmbeddings = embeddings
                    finalUsedPrompt = usedPrompt
                    wasPromptUpsampled = upsamplePrompt && (usedPrompt != enrichedPrompt)
                }
            }
            eval(textEmbeddings)

            // Classical CFG: encode the empty negative prompt with the same
            // encoder before unloading. Stored separately and consumed in the
            // denoising loop's uncond pass.
            if useClassicalCFG {
                Flux2Debug.log("Classical CFG enabled — encoding empty negative prompt")
                let negEmbeds: MLXArray
                switch model {
                case .dev:
                    negEmbeds = try textEncoder!.encode("", upsample: false)
                case .klein4B, .klein4BBase, .klein9B, .klein9BBase, .klein9BKV:
                    negEmbeds = try kleinEncoder!.encode("", upsample: false)
                }
                eval(negEmbeds)
                negativeTextEmbeddings = negEmbeds
                Flux2Debug.log("Negative embeddings shape: \(negEmbeds.shape)")
            }

            profiler.end("2. Text Encoding")

            Flux2Debug.log("Text embeddings shape: \(textEmbeddings.shape)")

            // Unload text encoder to free memory
            profiler.start("3. Unload Text Encoder")
            await unloadTextEncoder()
            profiler.end("3. Unload Text Encoder")
        }

        // === PHASE 2: Image Generation ===
        Flux2Debug.log("=== PHASE 2: Image Generation ===")

        // Check memory before loading transformer
        let phase2Check = memoryManager.checkImageGenerationPhase(config: quantization)
        if !phase2Check.isOk {
            Flux2Debug.log("Memory warning: \(phase2Check.message)")
        }

        // Load transformer
        profiler.start("4. Load Transformer")
        try await loadTransformer()
        profiler.end("4. Load Transformer")

        // Load VAE
        profiler.start("5. Load VAE")
        try await loadVAE()
        profiler.end("5. Load VAE")

        // Generate initial latents in PATCHIFIED format [B, 128, H/16, W/16]
        // This is the format expected by the BatchNorm normalization
        var patchifiedLatents: MLXArray

        switch mode {
        case .textToImage:
            patchifiedLatents = LatentUtils.generatePatchifiedLatents(
                height: validHeight,
                width: validWidth,
                seed: seed
            )
            Flux2Debug.log("Generated patchified latents: \(patchifiedLatents.shape)")

        case .strengthImageToImage(let initImage, _):
            // K4D LOCAL PATCH. VAE-encode the init image into the
            // same patchified-normalized format the existing pure-
            // noise T2I path produces — same shape, same
            // BatchNorm normalization, same packing later — so the
            // downstream denoise loop is unchanged. Noise mixing
            // happens AFTER the scheduler is set up (we need the
            // starting σ first); for now this is the CLEAN init in
            // latent space.
            guard let vae = vae else {
                throw Flux2Error.modelNotLoaded("VAE")
            }
            let processed = preprocessImageForVAE(
                initImage,
                targetHeight: validHeight,
                targetWidth: validWidth
            )
            // Deterministic mean encoding (samplePosterior=false) —
            // mirrors what encodeReferenceImages does; gives a
            // reproducible init latent rather than a noisy sample
            // from the posterior.
            let rawLatents = vae.encode(processed, samplePosterior: false)
            var initPatchified = LatentUtils.packLatentsToPatchified(rawLatents)
            initPatchified = LatentUtils.normalizeLatentsWithBatchNorm(
                initPatchified,
                runningMean: vae.batchNormRunningMean,
                runningVar: vae.batchNormRunningVar
            )
            eval(initPatchified)
            patchifiedLatents = initPatchified
            Flux2Debug.log("Encoded init image to patchified latents: \(patchifiedLatents.shape)")

        case .imageToImage(let images):
            // === FLUX.2 IMAGE-TO-IMAGE MODE ===
            // Flux.2 uses CONDITIONING mode for all I2I:
            // - Reference images are encoded and concatenated as context
            // - Output starts from random noise (full denoising)
            // - This is different from SD's "traditional" I2I with noise injection
            // Generate random noise for OUTPUT
            patchifiedLatents = LatentUtils.generatePatchifiedLatents(
                height: validHeight,
                width: validWidth,
                seed: seed
            )
            eval(patchifiedLatents)
            Flux2Debug.log("Generated output noise latents: \(patchifiedLatents.shape)")

            // Encode ALL reference images
            let (referenceLatents, referencePositionIds) = try encodeReferenceImages(
                images,
                height: validHeight,
                width: validWidth
            )
            eval(referenceLatents)
            Flux2Debug.log("Encoded \(images.count) reference images: latents \(referenceLatents.shape), posIds \(referencePositionIds.shape)")

            // Pack output latents to sequence format
            var packedOutputLatents = LatentUtils.packPatchifiedToSequence(patchifiedLatents)
            eval(packedOutputLatents)
            let outputSeqLen = packedOutputLatents.shape[1]
            Flux2Debug.log("Output sequence length: \(outputSeqLen)")

            // Generate position IDs for output and text
            let textLength = textEmbeddings.shape[1]
            let (textIds, outputImageIds, _) = LatentUtils.combinePositionIDs(
                textLength: textLength,
                height: validHeight,
                width: validWidth
            )

            // Concatenate reference latents to output latents for transformer input
            // [1, output_seq, 128] + [1, ref_seq, 128] -> [1, total_seq, 128]
            let combinedLatents = concatenated([packedOutputLatents, referenceLatents], axis: 1)
            Flux2Debug.log("Combined latents (output + refs): \(combinedLatents.shape)")
            _ = combinedLatents  // Used for documentation

            // Concatenate position IDs: output IDs + reference IDs
            // Output IDs are [output_seq, 4], reference IDs are [ref_seq, 4]
            let combinedImageIds = concatenated([outputImageIds, referencePositionIds], axis: 0)
            Flux2Debug.log("Combined image IDs: \(combinedImageIds.shape)")

            // Setup scheduler for FULL denoising (no timestep skip in Flux.2 I2I)
            // Check for custom sigmas from LoRA (e.g., Turbo LoRAs)
            if let customSigmas = loraSchedulerOverrides?.customSigmas {
                scheduler.setCustomSigmas(customSigmas)
            } else {
                scheduler.setTimesteps(numInferenceSteps: steps, imageSeqLen: outputSeqLen, strength: 1.0)
            }

            let effectiveSteps = scheduler.sigmas.count - 1
            Flux2Debug.log("Starting I2I denoising loop (\(effectiveSteps) steps)...")
            profiler.setTotalSteps(effectiveSteps)

            // Guidance tensor (nil for Klein models which don't use guidance embeddings)
            let guidanceTensor: MLXArray? = model.usesGuidanceEmbeds ? MLXArray([guidance]) : nil

            // MEMORY OPTIMIZATION: Set cache limit for denoising phase
            let phaseLimits = MemoryConfig.PhaseLimits.forModel(model, profile: memoryProfile)
            MemoryConfig.applyCacheLimit(bytes: phaseLimits.denoising)

            profiler.start("6. Denoising Loop")

            if model.supportsKVCache {
                // === KV-CACHED DENOISING PATH (klein-9b-kv) ===
                // Step 0: Extract KV cache from reference tokens
                // Steps 1+: Reuse cached KV (no reference re-processing, ~2.66x speedup)
                Flux2Debug.log("Using KV-cached denoising (\(effectiveSteps) steps, ~2.66x speedup)")

                guard let transformer = transformer else {
                    throw Flux2Error.generationCancelled
                }

                // Step 0: KV extraction pass
                let step0Start = Date()
                let sigma0 = scheduler.sigmas[0]
                let t0 = MLXArray([sigma0])

                let (noisePred0, kvCache) = transformer.forwardKVExtract(
                    hiddenStates: packedOutputLatents,
                    referenceHiddenStates: referenceLatents,
                    encoderHiddenStates: textEmbeddings,
                    timestep: t0,
                    guidance: guidanceTensor,
                    imgIds: outputImageIds,
                    txtIds: textIds,
                    refIds: referencePositionIds
                )

                packedOutputLatents = scheduler.step(
                    modelOutput: noisePred0,
                    timestep: sigma0,
                    sample: packedOutputLatents
                )
                eval(packedOutputLatents)

                if let onStep = onStep {
                    let sigmaNext = scheduler.sigmas[1]
                    let ctx = Flux2StepContext(
                        stepIdx: 0,
                        totalSteps: effectiveSteps,
                        sigma: sigma0,
                        sigmaNext: sigmaNext,
                        height: validHeight,
                        width: validWidth,
                        isI2I: true
                    )
                    packedOutputLatents = onStep(ctx, packedOutputLatents)
                    eval(packedOutputLatents)
                }

                let step0Duration = Date().timeIntervalSince(step0Start)
                profiler.recordStep(duration: step0Duration)
                onProgress?(1, effectiveSteps)
                Flux2Debug.log("Step 0 (KV extraction): \(String(format: "%.1f", step0Duration))s, cached \(kvCache.layerCount) layers")

                // Steps 1+: Cached denoising (no reference tokens in input)
                for stepIdx in 1..<(scheduler.sigmas.count - 1) {
                    let stepStart = Date()
                    let sigma = scheduler.sigmas[stepIdx]
                    let t = MLXArray([sigma])

                    let noisePred = transformer.forwardKVCached(
                        hiddenStates: packedOutputLatents,
                        encoderHiddenStates: textEmbeddings,
                        timestep: t,
                        guidance: guidanceTensor,
                        imgIds: outputImageIds,
                        txtIds: textIds,
                        kvCache: kvCache
                    )

                    packedOutputLatents = scheduler.step(
                        modelOutput: noisePred,
                        timestep: sigma,
                        sample: packedOutputLatents
                    )
                    eval(packedOutputLatents)

                    if let onStep = onStep {
                        let sigmaNext = scheduler.sigmas[stepIdx + 1]
                        let ctx = Flux2StepContext(
                            stepIdx: stepIdx,
                            totalSteps: effectiveSteps,
                            sigma: sigma,
                            sigmaNext: sigmaNext,
                            height: validHeight,
                            width: validWidth,
                            isI2I: true
                        )
                        packedOutputLatents = onStep(ctx, packedOutputLatents)
                        eval(packedOutputLatents)
                    }

                    if clearCacheEveryNSteps > 0 && (stepIdx + 1) % clearCacheEveryNSteps == 0 {
                        MemoryConfig.clearCache()
                    }

                    let stepDuration = Date().timeIntervalSince(stepStart)
                    profiler.recordStep(duration: stepDuration)
                    onProgress?(stepIdx + 1, effectiveSteps)
                    Flux2Debug.verbose("Step \(stepIdx + 1)/\(effectiveSteps) (cached)")

                    // Checkpoint
                    if let interval = checkpointInterval,
                       let checkpointCallback = onCheckpoint,
                       (stepIdx + 1) % interval == 0 {
                        var checkpointPatchified = LatentUtils.unpackSequenceToPatchified(
                            packedOutputLatents,
                            height: validHeight,
                            width: validWidth
                        )
                        checkpointPatchified = LatentUtils.denormalizeLatentsWithBatchNorm(
                            checkpointPatchified,
                            runningMean: vae!.batchNormRunningMean,
                            runningVar: vae!.batchNormRunningVar
                        )
                        let checkpointLatents = LatentUtils.unpatchifyLatents(checkpointPatchified)
                        eval(checkpointLatents)

                        let checkpointDecoded = vae!.decode(checkpointLatents)
                        eval(checkpointDecoded)

                        if let checkpointImage = postprocessVAEOutput(checkpointDecoded) {
                            checkpointCallback(stepIdx + 1, checkpointImage)
                        }
                    }
                }

                // KV cache is freed when it goes out of scope
                Flux2Debug.log("KV-cached denoising complete")

            } else {
            // === STANDARD I2I DENOISING PATH ===

            // Classical CFG: precompute uncond textIds once.
            let i2iUncondTextIds: MLXArray?
            if useClassicalCFG, let neg = negativeTextEmbeddings {
                i2iUncondTextIds = LatentUtils.generateTextPositionIDs(length: neg.shape[1])
                Flux2Debug.log("Classical CFG active for I2I — guidance=\(guidance), uncond seq=\(neg.shape[1])")
            } else {
                i2iUncondTextIds = nil
            }

            for stepIdx in 0..<(scheduler.sigmas.count - 1) {
                let stepStart = Date()

                let sigma = scheduler.sigmas[stepIdx]
                let t = MLXArray([sigma])

                // Concatenate current output latents with reference latents for this step
                let inputLatents = concatenated([packedOutputLatents, referenceLatents], axis: 1)

                // Check transformer is still loaded (may be unloaded during cancellation)
                guard let transformer = transformer else {
                    throw Flux2Error.generationCancelled
                }

                // K4D LOCAL PATCH — per-block reference K/V strength (Phase D, 2026-05-20).
                // Build the context once per step. outputSeqLen is the
                // main_img token count (the ref tokens start right after).
                // Skip when strength == 1.0 (no-op anyway).
                let refStrengthCtx: Flux2RefScalingContext? = (referenceStrength != 1.0)
                    ? Flux2RefScalingContext(refTokenStartInImage: outputSeqLen, strength: referenceStrength)
                    : nil

                // Run transformer (conditional pass)
                let noisePredCond = transformer.callAsFunction(
                    hiddenStates: inputLatents,
                    encoderHiddenStates: textEmbeddings,
                    timestep: t,
                    guidance: guidanceTensor,
                    imgIds: combinedImageIds,
                    txtIds: textIds,
                    refScaling: refStrengthCtx
                )

                // Classical CFG: second (unconditional) pass with the SAME image
                // conditioning — reference latents stay, only the text changes.
                let noisePred: MLXArray
                if useClassicalCFG,
                   let negEmbeds = negativeTextEmbeddings,
                   let uncondIds = i2iUncondTextIds {
                    let noisePredUncond = transformer.callAsFunction(
                        hiddenStates: inputLatents,
                        encoderHiddenStates: negEmbeds,
                        timestep: t,
                        guidance: guidanceTensor,
                        imgIds: combinedImageIds,
                        txtIds: uncondIds,
                        refScaling: refStrengthCtx
                    )
                    noisePred = noisePredUncond + MLXArray(guidance) * (noisePredCond - noisePredUncond)
                } else {
                    noisePred = noisePredCond
                }

                // Extract only the OUTPUT portion of the noise prediction
                // noisePred shape is [1, total_seq, 128], we want first outputSeqLen
                let outputNoisePred = noisePred[0..., 0..<outputSeqLen, 0...]

                // Scheduler step on OUTPUT latents only
                packedOutputLatents = scheduler.step(
                    modelOutput: outputNoisePred,
                    timestep: sigma,
                    sample: packedOutputLatents
                )
                eval(packedOutputLatents)

                // Step hook (see T2I path for rationale on running every step).
                if let onStep = onStep {
                    let sigmaNext = scheduler.sigmas[stepIdx + 1]
                    let ctx = Flux2StepContext(
                        stepIdx: stepIdx,
                        totalSteps: effectiveSteps,
                        sigma: sigma,
                        sigmaNext: sigmaNext,
                        height: validHeight,
                        width: validWidth,
                        isI2I: true
                    )
                    packedOutputLatents = onStep(ctx, packedOutputLatents)
                    eval(packedOutputLatents)
                }

                // MEMORY OPTIMIZATION: Periodic cache clearing to prevent memory buildup
                if clearCacheEveryNSteps > 0 && (stepIdx + 1) % clearCacheEveryNSteps == 0 {
                    MemoryConfig.clearCache()
                }

                let stepDuration = Date().timeIntervalSince(stepStart)
                profiler.recordStep(duration: stepDuration)

                onProgress?(stepIdx + 1, effectiveSteps)
                Flux2Debug.verbose("Step \(stepIdx + 1)/\(effectiveSteps)")

                // Checkpoint
                if let interval = checkpointInterval,
                   let checkpointCallback = onCheckpoint,
                   (stepIdx + 1) % interval == 0 {
                    var checkpointPatchified = LatentUtils.unpackSequenceToPatchified(
                        packedOutputLatents,
                        height: validHeight,
                        width: validWidth
                    )
                    checkpointPatchified = LatentUtils.denormalizeLatentsWithBatchNorm(
                        checkpointPatchified,
                        runningMean: vae!.batchNormRunningMean,
                        runningVar: vae!.batchNormRunningVar
                    )
                    let checkpointLatents = LatentUtils.unpatchifyLatents(checkpointPatchified)
                    eval(checkpointLatents)

                    let checkpointDecoded = vae!.decode(checkpointLatents)
                    eval(checkpointDecoded)

                    if let checkpointImage = postprocessVAEOutput(checkpointDecoded) {
                        checkpointCallback(stepIdx + 1, checkpointImage)
                    }
                }

                if stepIdx % 10 == 0 {
                    memoryManager.clearCache()
                }
            }

            } // end else (standard I2I path)

            profiler.end("6. Denoising Loop")

            // Decode final OUTPUT latents
            profiler.start("7. VAE Decode")
            var finalPatchified = LatentUtils.unpackSequenceToPatchified(
                packedOutputLatents,
                height: validHeight,
                width: validWidth
            )
            finalPatchified = LatentUtils.denormalizeLatentsWithBatchNorm(
                finalPatchified,
                runningMean: vae!.batchNormRunningMean,
                runningVar: vae!.batchNormRunningVar
            )
            let finalLatents = LatentUtils.unpatchifyLatents(finalPatchified)
            eval(finalLatents)

            let decoded = vae!.decode(finalLatents)
            eval(decoded)
            profiler.end("7. VAE Decode")

            profiler.start("8. Post-processing")
            guard let image = postprocessVAEOutput(decoded) else {
                throw Flux2Error.generationFailed("Failed to convert VAE output to image")
            }
            profiler.end("8. Post-processing")

            if profiler.isEnabled {
                print(profiler.generateReport())
            }

            return Flux2GenerationResult(
                image: image,
                usedPrompt: finalUsedPrompt,
                wasUpsampled: wasPromptUpsampled,
                originalPrompt: prompt
            )

        // ============================================================
        // K4D LOCAL PATCH — strength img2img + Kontext refs handler
        // ============================================================
        // Self-contained, parallels the standard (non-KV) `.imageToImage`
        // path above but starts from a VAE-encoded init image (with a
        // FlowMatch noise blend at the strength's starting sigma)
        // instead of pure noise. The denoise loop concatenates the
        // refs onto the output latents at every step exactly like the
        // standard I2I path, and slices the noise prediction back to
        // the output portion before the scheduler step. Refs stay
        // clean across all steps.
        //
        // ROLLBACK: delete this entire `case` block. The
        // `.strengthImageToImageWithRefs` enum case can stay or be
        // removed depending on whether the wrapper still references it.
        case .strengthImageToImageWithRefs(let initImage, let strength, let refImages):
            guard let vae = vae else {
                throw Flux2Error.modelNotLoaded("VAE")
            }

            // ---- 1. VAE-encode init → patchified normalized latents ----
            let processedInit = preprocessImageForVAE(
                initImage,
                targetHeight: validHeight,
                targetWidth: validWidth
            )
            let rawInitLatents = vae.encode(processedInit, samplePosterior: false)
            var initPatchified = LatentUtils.packLatentsToPatchified(rawInitLatents)
            initPatchified = LatentUtils.normalizeLatentsWithBatchNorm(
                initPatchified,
                runningMean: vae.batchNormRunningMean,
                runningVar: vae.batchNormRunningVar
            )
            eval(initPatchified)
            Flux2Debug.log("[strength+refs] Encoded init to patchified: \(initPatchified.shape)")

            // ---- 2. Encode all Kontext reference images ----
            let (referenceLatents, referencePositionIds) = try encodeReferenceImages(
                refImages,
                height: validHeight,
                width: validWidth
            )
            eval(referenceLatents)
            Flux2Debug.log("[strength+refs] Encoded \(refImages.count) refs: latents \(referenceLatents.shape), posIds \(referencePositionIds.shape)")

            // ---- 3. Compute the output sequence length (needed for
            // the scheduler's mu calculation). The latents are still
            // in patchified shape at this point so the noise blend
            // happens in canonical [B, 128, H/16, W/16] layout —
            // see the noise-layout comment below.
            let h16 = validHeight / 16
            let w16 = validWidth / 16
            let outputSeqLen = h16 * w16

            // ---- 4. Generate combined position IDs (output + refs) ----
            let textLength = textEmbeddings.shape[1]
            let (textIds, outputImageIds, _) = LatentUtils.combinePositionIDs(
                textLength: textLength,
                height: validHeight,
                width: validWidth
            )
            let combinedImageIds = concatenated([outputImageIds, referencePositionIds], axis: 0)
            Flux2Debug.log("[strength+refs] Combined image IDs: \(combinedImageIds.shape)")

            // ---- 5. Setup scheduler with strength (partial trajectory) ----
            if let customSigmas = loraSchedulerOverrides?.customSigmas {
                scheduler.setCustomSigmas(customSigmas)
            } else {
                scheduler.setTimesteps(numInferenceSteps: steps, imageSeqLen: outputSeqLen, strength: strength)
            }

            // ---- 6. FlowMatch noise blend in PATCHIFIED shape, then pack ----
            //   x_start = (1 - σ_start) * init + σ_start * noise
            //
            // CRITICAL BUG FIX 2026-05-17: this blend MUST happen in
            // the canonical patchified shape [B, 128, H/16, W/16],
            // not in the post-pack sequence shape [B, H/16*W/16, 128].
            // MLX's `MLXRandom.normal(shape)` fills memory in C-order
            // for whatever shape it's given — so generating noise as
            // [1, 4096, 128] produces a DIFFERENT per-position value
            // distribution than generating as [1, 128, 64, 64] and
            // then packing. Klein's transformer treats the noise as
            // a structured spatial latent, and the wrong per-position
            // layout silently degrades attention to Kontext refs.
            // The original broken version produced no donor influence;
            // this version matches the standard I2I path's canonical
            // noise layout and refs attend correctly.
            //
            // Refs are NEVER noise-blended; they remain clean
            // conditioning across all denoise steps.
            let startSigma = scheduler.sigmas[0]
            let noisePatchified = MLXRandom.normal([1, 128, h16, w16])
            eval(noisePatchified)
            let blendedPatchified = MLXArray(1 - startSigma) * initPatchified + MLXArray(startSigma) * noisePatchified
            eval(blendedPatchified)
            var packedOutputLatents = LatentUtils.packPatchifiedToSequence(blendedPatchified)
            eval(packedOutputLatents)
            Flux2Debug.log("[strength+refs] Noise blend (patchified): σ_start=\(startSigma), strength=\(strength)")

            let effectiveSteps = scheduler.sigmas.count - 1
            Flux2Debug.log("[strength+refs] Starting denoising loop (\(effectiveSteps) steps)")
            profiler.setTotalSteps(effectiveSteps)

            // ---- 7. Denoise loop with refs concatenated each step ----
            let guidanceTensorR: MLXArray? = model.usesGuidanceEmbeds ? MLXArray([guidance]) : nil
            MemoryConfig.applyCacheLimit(bytes: phaseLimits.denoising)
            profiler.start("6. Denoising Loop")

            for stepIdx in 0..<(scheduler.sigmas.count - 1) {
                let stepStart = Date()
                let sigma = scheduler.sigmas[stepIdx]
                let tArr = MLXArray([sigma])

                let inputLatents = concatenated([packedOutputLatents, referenceLatents], axis: 1)

                guard let transformer = transformer else {
                    throw Flux2Error.generationCancelled
                }

                // K4D LOCAL PATCH — per-block reference K/V strength (Phase D).
                let refStrengthCtxR: Flux2RefScalingContext? = (referenceStrength != 1.0)
                    ? Flux2RefScalingContext(refTokenStartInImage: outputSeqLen, strength: referenceStrength)
                    : nil

                let noisePred = transformer.callAsFunction(
                    hiddenStates: inputLatents,
                    encoderHiddenStates: textEmbeddings,
                    timestep: tArr,
                    guidance: guidanceTensorR,
                    imgIds: combinedImageIds,
                    txtIds: textIds,
                    refScaling: refStrengthCtxR
                )

                // Slice noise prediction to output portion only —
                // refs are static conditioning, never get scheduled.
                let outputNoisePred = noisePred[0..., 0..<outputSeqLen, 0...]

                packedOutputLatents = scheduler.step(
                    modelOutput: outputNoisePred,
                    timestep: sigma,
                    sample: packedOutputLatents
                )
                eval(packedOutputLatents)

                if clearCacheEveryNSteps > 0 && (stepIdx + 1) % clearCacheEveryNSteps == 0 {
                    MemoryConfig.clearCache()
                }

                let stepDuration = Date().timeIntervalSince(stepStart)
                profiler.recordStep(duration: stepDuration)
                onProgress?(stepIdx + 1, effectiveSteps)
                Flux2Debug.verbose("[strength+refs] Step \(stepIdx + 1)/\(effectiveSteps)")

                if let interval = checkpointInterval,
                   let checkpointCallback = onCheckpoint,
                   (stepIdx + 1) % interval == 0 {
                    var checkpointPatchified = LatentUtils.unpackSequenceToPatchified(
                        packedOutputLatents,
                        height: validHeight,
                        width: validWidth
                    )
                    checkpointPatchified = LatentUtils.denormalizeLatentsWithBatchNorm(
                        checkpointPatchified,
                        runningMean: vae.batchNormRunningMean,
                        runningVar: vae.batchNormRunningVar
                    )
                    let checkpointLatents = LatentUtils.unpatchifyLatents(checkpointPatchified)
                    eval(checkpointLatents)
                    let checkpointDecoded = vae.decode(checkpointLatents)
                    eval(checkpointDecoded)
                    if let checkpointImage = postprocessVAEOutput(checkpointDecoded) {
                        checkpointCallback(stepIdx + 1, checkpointImage)
                    }
                }

                if stepIdx % 10 == 0 {
                    memoryManager.clearCache()
                }
            }
            profiler.end("6. Denoising Loop")

            // ---- 8. Decode final output latents ----
            profiler.start("7. VAE Decode")
            var finalPatchifiedR = LatentUtils.unpackSequenceToPatchified(
                packedOutputLatents,
                height: validHeight,
                width: validWidth
            )
            finalPatchifiedR = LatentUtils.denormalizeLatentsWithBatchNorm(
                finalPatchifiedR,
                runningMean: vae.batchNormRunningMean,
                runningVar: vae.batchNormRunningVar
            )
            let finalLatentsR = LatentUtils.unpatchifyLatents(finalPatchifiedR)
            eval(finalLatentsR)

            let decodedR = vae.decode(finalLatentsR)
            eval(decodedR)
            profiler.end("7. VAE Decode")

            profiler.start("8. Post-processing")
            guard let imageR = postprocessVAEOutput(decodedR) else {
                throw Flux2Error.generationFailed("Failed to convert VAE output to image")
            }
            profiler.end("8. Post-processing")

            if profiler.isEnabled {
                print(profiler.generateReport())
            }

            return Flux2GenerationResult(
                image: imageR,
                usedPrompt: finalUsedPrompt,
                wasUpsampled: wasPromptUpsampled,
                originalPrompt: prompt
            )
        // END K4D LOCAL PATCH — strength img2img + refs handler
        }

        // === TEXT-TO-IMAGE PATH (I2I returns earlier) ===

        // Pack patchified latents to sequence format for transformer [B, seq_len, 128]
        var packedLatents = LatentUtils.packPatchifiedToSequence(patchifiedLatents)
        eval(packedLatents)

        // Generate position IDs for latents and text
        let textLength = textEmbeddings.shape[1]
        let (textIds, imageIds, _) = LatentUtils.combinePositionIDs(
            textLength: textLength,
            height: validHeight,
            width: validWidth
        )

        // Calculate image sequence length for scheduler mu
        let imageSeqLen = packedLatents.shape[1]
        Flux2Debug.log("Image sequence length: \(imageSeqLen)")

        // Setup scheduler.
        //
        // K4D LOCAL PATCH: extract strength from the mode so the
        // .strengthImageToImage path can request a partial trajectory
        // (e.g. strength=0.7 → start denoising from sigma≈0.7
        // instead of 1.0, skipping the early-noise steps). For
        // .textToImage and any future case, the default 1.0 keeps the
        // pre-patch behavior bit-identical.
        let effectiveStrength: Float = {
            if case .strengthImageToImage(_, let s) = mode { return s }
            return 1.0
        }()
        if let customSigmas = loraSchedulerOverrides?.customSigmas {
            scheduler.setCustomSigmas(customSigmas)
        } else {
            scheduler.setTimesteps(numInferenceSteps: steps, imageSeqLen: imageSeqLen, strength: effectiveStrength)
        }

        // K4D LOCAL PATCH: FlowMatch noise injection for strength
        // img2img. After the scheduler has resolved its (possibly
        // truncated) sigma schedule, blend the CLEAN init latent
        // with fresh noise at the starting sigma:
        //
        //     x_start = (1 - σ_start) * init + σ_start * noise
        //
        // This is the rectified-flow analog of SD's "noise at the
        // strength's timestep." With strength=1.0, σ_start=1.0 and
        // the formula reduces to pure noise — regression-safe.
        if case .strengthImageToImage = mode {
            let startSigma = scheduler.sigmas[0]
            let noise = MLXRandom.normal(packedLatents.shape)
            packedLatents = MLXArray(1 - startSigma) * packedLatents + MLXArray(startSigma) * noise
            eval(packedLatents)
            Flux2Debug.log("Strength-img2img noise blend: σ_start=\(startSigma), strength=\(effectiveStrength)")
        }

        let effectiveSteps = scheduler.sigmas.count - 1
        Flux2Debug.log("Starting denoising loop (\(effectiveSteps) steps)...")
        profiler.setTotalSteps(effectiveSteps)

        // OPTIMIZATION: Create guidance tensor ONCE before the loop
        // Klein models don't use guidance embeddings
        let guidanceTensor: MLXArray? = model.usesGuidanceEmbeds ? MLXArray([guidance]) : nil

        // Classical CFG (klein base): precompute uncond text IDs once.
        let uncondTextIds: MLXArray?
        if useClassicalCFG, let neg = negativeTextEmbeddings {
            uncondTextIds = LatentUtils.generateTextPositionIDs(length: neg.shape[1])
            Flux2Debug.log("Classical CFG active — guidance=\(guidance), uncond seq=\(neg.shape[1])")
        } else {
            uncondTextIds = nil
        }

        // MEMORY OPTIMIZATION: Set cache limit for denoising phase
        MemoryConfig.applyCacheLimit(bytes: phaseLimits.denoising)

        profiler.start("6. Denoising Loop")

        // Denoising loop - use sigmas (in [0, 1] range) for transformer
        for stepIdx in 0..<(scheduler.sigmas.count - 1) {
            let stepStart = Date()

            let sigma = scheduler.sigmas[stepIdx]
            let t = MLXArray([sigma])

            // Check transformer is still loaded (may be unloaded during cancellation)
            guard let transformer = transformer else {
                throw Flux2Error.generationCancelled
            }

            // Run transformer (conditional pass)
            let noisePredCond = transformer.callAsFunction(
                hiddenStates: packedLatents,
                encoderHiddenStates: textEmbeddings,
                timestep: t,
                guidance: guidanceTensor,
                imgIds: imageIds,
                txtIds: textIds
            )

            // Classical CFG: second (unconditional) pass + linear combination.
            // Matches diffusers' `noise = uncond + guidance * (cond - uncond)`.
            let noisePred: MLXArray
            if useClassicalCFG,
               let negEmbeds = negativeTextEmbeddings,
               let uncondIds = uncondTextIds {
                let noisePredUncond = transformer.callAsFunction(
                    hiddenStates: packedLatents,
                    encoderHiddenStates: negEmbeds,
                    timestep: t,
                    guidance: guidanceTensor,
                    imgIds: imageIds,
                    txtIds: uncondIds
                )
                noisePred = noisePredUncond + MLXArray(guidance) * (noisePredCond - noisePredUncond)
            } else {
                noisePred = noisePredCond
            }

            // Scheduler step
            packedLatents = scheduler.step(
                modelOutput: noisePred,
                timestep: sigma,
                sample: packedLatents
            )

            // Synchronize GPU
            eval(packedLatents)

            // Step hook (e.g. RePaint masked-inpainting blend). Runs every step
            // including the final one — chains that rely on `sigmaNext == 0` to
            // restore the unchanged region must see that final iteration.
            if let onStep = onStep {
                let sigmaNext = scheduler.sigmas[stepIdx + 1]
                let ctx = Flux2StepContext(
                    stepIdx: stepIdx,
                    totalSteps: effectiveSteps,
                    sigma: sigma,
                    sigmaNext: sigmaNext,
                    height: validHeight,
                    width: validWidth,
                    isI2I: false
                )
                packedLatents = onStep(ctx, packedLatents)
                eval(packedLatents)
            }

            // MEMORY OPTIMIZATION: Periodic cache clearing to prevent memory buildup
            if clearCacheEveryNSteps > 0 && (stepIdx + 1) % clearCacheEveryNSteps == 0 {
                MemoryConfig.clearCache()
            }

            // Record step time
            let stepDuration = Date().timeIntervalSince(stepStart)
            profiler.recordStep(duration: stepDuration)

            // Report progress (using effective steps for I2I)
            onProgress?(stepIdx + 1, effectiveSteps)

            Flux2Debug.verbose("Step \(stepIdx + 1)/\(effectiveSteps)")

            // Generate checkpoint image if requested
            if let interval = checkpointInterval,
               let checkpointCallback = onCheckpoint,
               (stepIdx + 1) % interval == 0 {
                Flux2Debug.verbose("Generating checkpoint at step \(stepIdx + 1)...")

                // Decode current latents to image
                var checkpointPatchified = LatentUtils.unpackSequenceToPatchified(
                    packedLatents,
                    height: validHeight,
                    width: validWidth
                )
                checkpointPatchified = LatentUtils.denormalizeLatentsWithBatchNorm(
                    checkpointPatchified,
                    runningMean: vae!.batchNormRunningMean,
                    runningVar: vae!.batchNormRunningVar
                )
                let checkpointLatents = LatentUtils.unpatchifyLatents(checkpointPatchified)
                eval(checkpointLatents)

                let checkpointDecoded = vae!.decode(checkpointLatents)
                eval(checkpointDecoded)
                Flux2Debug.verbose("Checkpoint VAE output shape: \(checkpointDecoded.shape)")

                if let checkpointImage = postprocessVAEOutput(checkpointDecoded) {
                    checkpointCallback(stepIdx + 1, checkpointImage)
                } else {
                    Flux2Debug.log("Warning: Failed to convert checkpoint to image at step \(stepIdx + 1)")
                }
            }

            // Periodic memory cleanup
            if stepIdx % 10 == 0 {
                memoryManager.clearCache()
            }
        }

        profiler.end("6. Denoising Loop")

        Flux2Debug.log("Denoising complete, decoding image...")

        // Unpack sequence latents back to patchified format [B, 128, H/16, W/16]
        var patchifiedFinal = LatentUtils.unpackSequenceToPatchified(
            packedLatents,
            height: validHeight,
            width: validWidth
        )
        Flux2Debug.log("Unpacked patchified latents: \(patchifiedFinal.shape)")

        // CRITICAL: Denormalize patchified latents with VAE BatchNorm AFTER denoising
        // This reverses the normalization applied before the transformer
        Flux2Debug.log("Denormalizing patchified latents with BatchNorm...")
        patchifiedFinal = LatentUtils.denormalizeLatentsWithBatchNorm(
            patchifiedFinal,
            runningMean: vae!.batchNormRunningMean,
            runningVar: vae!.batchNormRunningVar
        )
        eval(patchifiedFinal)

        // Unpatchify to VAE format [B, 32, H/8, W/8]
        let finalLatents = LatentUtils.unpatchifyLatents(patchifiedFinal)
        Flux2Debug.log("Final latents for VAE: \(finalLatents.shape)")
        eval(finalLatents)

        // === PHASE 3: Decode to Image ===
        Flux2Debug.log("=== PHASE 3: VAE Decoding ===")

        // MEMORY OPTIMIZATION: Set cache limit for VAE decoding phase
        MemoryConfig.applyCacheLimit(bytes: phaseLimits.vaeDecoding)

        profiler.start("7. VAE Decode")
        let decoded = vae!.decode(finalLatents)
        eval(decoded)
        profiler.end("7. VAE Decode")

        // Convert to CGImage
        profiler.start("8. Post-processing")
        guard let image = postprocessVAEOutput(decoded) else {
            throw Flux2Error.imageProcessingFailed("Failed to convert output to image")
        }
        profiler.end("8. Post-processing")

        Flux2Debug.log("Generation complete!")
        memoryManager.logMemoryState()

        // Print profiling report if enabled
        if profiler.isEnabled {
            print(profiler.generateReport())
        }

        return Flux2GenerationResult(
            image: image,
            usedPrompt: finalUsedPrompt,
            wasUpsampled: wasPromptUpsampled,
            originalPrompt: prompt
        )
    }

    // MARK: - Private Methods

    /// Encode reference images for image-to-image generation (Flux.2 conditioning mode)
    ///
    /// Following the reference Flux.2 diffusers implementation:
    /// 1. Each image is encoded SEPARATELY through VAE
    /// 2. Each image gets a UNIQUE T-coordinate (T=10, T=20, T=30, etc.)
    /// 3. Latents are concatenated AFTER encoding
    ///
    /// This preserves semantic information:
    /// - The transformer can distinguish between different reference images
    /// - Each reference image has its own position in the "time" dimension
    /// - Images are resized to max 512² pixels each for memory efficiency
    ///
    /// - Parameters:
    ///   - images: Reference images (1-10 supported)
    ///   - height: Target output height
    ///   - width: Target output width
    /// - Returns: Tuple of (latents [1, seq_len, 128], position IDs [seq_len, 4])
    private func encodeReferenceImages(
        _ images: [CGImage],
        height: Int,
        width: Int
    ) throws -> (latents: MLXArray, positionIds: MLXArray) {
        guard let vae = vae else {
            throw Flux2Error.modelNotLoaded("VAE")
        }

        guard !images.isEmpty else {
            throw Flux2Error.invalidConfiguration("No reference images provided")
        }

        Flux2Debug.log("Encoding \(images.count) reference images separately with unique T-coordinates...")

        // === STEP 1: Process each image separately ===
        // Max area per image - matches diffusers pipeline_flux2.py line 892-893
        // Reference uses 1024² for conditioning images (not 768² which is for upsampling)
        let maxImageArea = 1024 * 1024  // ~4096 tokens per image
        let multipleOf = 32  // vae_scale_factor * 2

        var allPackedLatents: [MLXArray] = []
        var latentHeights: [Int] = []
        var latentWidths: [Int] = []

        for (index, image) in images.enumerated() {
            Flux2Debug.log("Processing reference image \(index + 1)/\(images.count): \(image.width)x\(image.height)")

            // Calculate target dimensions for this image
            var targetWidth = image.width
            var targetHeight = image.height
            let pixelCount = targetWidth * targetHeight

            if pixelCount > maxImageArea {
                let scale = sqrt(Double(maxImageArea) / Double(pixelCount))
                targetWidth = Int(Double(targetWidth) * scale)
                targetHeight = Int(Double(targetHeight) * scale)
            }

            // Make dimensions multiples of 32
            targetWidth = (targetWidth / multipleOf) * multipleOf
            targetHeight = (targetHeight / multipleOf) * multipleOf

            // Ensure minimum size
            targetWidth = max(targetWidth, multipleOf)
            targetHeight = max(targetHeight, multipleOf)

            Flux2Debug.log("  -> Resized to \(targetWidth)x\(targetHeight)")

            // Preprocess and encode
            let processed = preprocessImageForVAE(image, targetHeight: targetHeight, targetWidth: targetWidth)

            // Encode with VAE -> [1, 32, H/8, W/8]
            // IMPORTANT: Use samplePosterior=false to get deterministic mean (like diffusers argmax)
            // This preserves color information better than sampling with noise
            let rawLatents = vae.encode(processed, samplePosterior: false)

            // Patchify: [1, 32, H/8, W/8] -> [1, 128, H/16, W/16]
            var patchified = LatentUtils.packLatentsToPatchified(rawLatents)

            // Normalize with BatchNorm (critical for Flux.2)
            patchified = LatentUtils.normalizeLatentsWithBatchNorm(
                patchified,
                runningMean: vae.batchNormRunningMean,
                runningVar: vae.batchNormRunningVar
            )
            eval(patchified)

            // K4D LOCAL PATCH — reference channel mask (2026-05-17).
            // patchified shape: [1, 128, H/16, W/16]. Channels 0-63 carry
            // structure/layout; channels 64-127 carry texture/detail (per
            // capitan01R's hook-tracing of Klein 9B). Zero one half if
            // the user has requested structure-only or texture-only donor
            // contribution. `.all` is the default no-op.
            if referenceChannelMask != .all {
                let h16 = patchified.shape[2]
                let w16 = patchified.shape[3]
                let zeros64 = MLXArray.zeros([1, 64, h16, w16], dtype: patchified.dtype)
                switch referenceChannelMask {
                case .all:
                    break // already handled by outer guard
                case .low:
                    // Keep channels 0-63 (structure), zero out 64-127 (texture).
                    let lowChans = patchified[0..., 0..<64, 0..., 0...]
                    patchified = concatenated([lowChans, zeros64], axis: 1)
                case .high:
                    // Zero channels 0-63 (structure), keep 64-127 (texture).
                    let highChans = patchified[0..., 64..<128, 0..., 0...]
                    patchified = concatenated([zeros64, highChans], axis: 1)
                }
                eval(patchified)
                Flux2Debug.log("  -> Channel mask applied: \(referenceChannelMask.rawValue) (ref \(index + 1))")
            }
            // END K4D LOCAL PATCH — reference channel mask

            // K4D LOCAL PATCH — parametric reference spatial fade (Phase 2.3, 2026-05-20).
            // Build a [H/16, W/16] multiplier mask from the parametric
            // (origin, shape, radius, floor) inputs and apply to the
            // patchified ref latent's 128 channels via broadcast multiply.
            //
            // Math at each (y, x):
            //   normY = (y + 0.5) / h16,  normX = (x + 0.5) / w16    (canvas-normalized)
            //   distance = (shape-dependent — see switch below)
            //   normDist = clamp(distance / radius, 0, 1)
            //   strength = 1.0 + (floor - 1.0) * normDist
            //     → 1.0 at origin; linearly to `floor` at `radius`; floor beyond
            if referenceSpatialFade.shape != .off {
                let h16 = patchified.shape[2]
                let w16 = patchified.shape[3]
                let originX = referenceSpatialFade.originX
                let originY = referenceSpatialFade.originY
                let radius  = referenceSpatialFade.radius
                let floor   = referenceSpatialFade.floor

                // Per-axis grid of canvas-normalized coordinates.
                // Use pixel-center sampling (+0.5) so the multiplier at
                // (originX, originY) reaches the documented 1.0 max
                // at the row/col closest to the requested origin.
                let hIdx = MLXArray.arange(h16, dtype: .float32)
                let wIdx = MLXArray.arange(w16, dtype: .float32)
                let normY = (hIdx + MLXArray(Float(0.5))) / Float(h16)  // [H]
                let normX = (wIdx + MLXArray(Float(0.5))) / Float(w16)  // [W]

                // Distance grid in canvas-normalized units.
                let distance: MLXArray
                switch referenceSpatialFade.shape {
                case .off:
                    // unreachable thanks to outer guard
                    distance = MLXArray.zeros([h16, w16], dtype: .float32)
                case .horizontal:
                    // Only x-component of distance.
                    let dx = MLX.abs(normX - MLXArray(originX))                       // [W]
                    distance = MLX.broadcast(dx.expandedDimensions(axis: 0), to: [h16, w16])
                case .vertical:
                    // Only y-component.
                    let dy = MLX.abs(normY - MLXArray(originY))                       // [H]
                    distance = MLX.broadcast(dy.expandedDimensions(axis: 1), to: [h16, w16])
                case .radial:
                    // Euclidean distance from origin.
                    let dy = normY - MLXArray(originY)                                 // [H]
                    let dx = normX - MLXArray(originX)                                 // [W]
                    let dyB = MLX.broadcast(dy.expandedDimensions(axis: 1), to: [h16, w16])
                    let dxB = MLX.broadcast(dx.expandedDimensions(axis: 0), to: [h16, w16])
                    distance = MLX.sqrt(dyB * dyB + dxB * dxB)
                }

                // Clamp distance/radius to [0, 1] then linearly ramp to floor.
                let normDist = MLX.minimum(distance / Float(radius), MLXArray(Float(1.0)))
                let multiplier = MLXArray(Float(1.0)) + normDist * Float(floor - 1.0)  // [H, W]

                // Broadcast to [1, 1, H, W] for channel-broadcast multiply.
                let mask4d = multiplier.expandedDimensions(axes: [0, 1])
                patchified = patchified * mask4d
                eval(patchified)
                Flux2Debug.log(
                    "  -> Spatial fade applied: shape=\(referenceSpatialFade.shape.rawValue) " +
                    "origin=(\(originX),\(originY)) radius=\(radius) floor=\(floor) (ref \(index + 1))"
                )
            }
            // END K4D LOCAL PATCH — parametric reference spatial fade

            // Pack to sequence: [1, 128, H/16, W/16] -> [1, seq_len, 128]
            let packedLatents = LatentUtils.packPatchifiedToSequence(patchified)

            // Remove batch dimension for concatenation: [1, seq_len, 128] -> [seq_len, 128]
            let squeezed = packedLatents.squeezed(axis: 0)
            allPackedLatents.append(squeezed)

            // Track dimensions for position IDs
            let patchifiedH = targetHeight / 16
            let patchifiedW = targetWidth / 16
            latentHeights.append(patchifiedH)
            latentWidths.append(patchifiedW)

            let seqLen = squeezed.shape[0]
            Flux2Debug.log("  -> Encoded to \(seqLen) tokens (T-coordinate will be \(10 + 10 * index))")
        }

        // === STEP 2: Concatenate all latents ===
        // Stack along sequence dimension: [seq1, 128] + [seq2, 128] + ... -> [total_seq, 128]
        let concatenatedLatents = concatenated(allPackedLatents, axis: 0)

        // Add batch dimension back: [total_seq, 128] -> [1, total_seq, 128]
        let finalLatents = concatenatedLatents.expandedDimensions(axis: 0)

        let totalSeqLen = finalLatents.shape[1]
        Flux2Debug.log("Combined reference latents: \(finalLatents.shape) (total \(totalSeqLen) tokens)")

        // === STEP 3: Generate position IDs with unique T-coordinates per image ===
        let positionIds = LatentUtils.generateReferenceImagePositionIDs(
            latentHeights: latentHeights,
            latentWidths: latentWidths,
            scale: 10
        )

        Flux2Debug.log("Position IDs generated: \(positionIds.shape)")
        Flux2Debug.log("Reference encoding complete: \(images.count) images with unique T-coordinates")

        return (latents: finalLatents, positionIds: positionIds)
    }

    /// Create CGImage from raw image data (PNG/JPEG) using CGImageSource for pixel-exact decoding.
    /// This avoids NSImage roundtrip which can introduce subpixel shifts via AppKit re-rendering.
    public static func cgImage(from data: Data) -> CGImage? {
        guard let source = CGImageSourceCreateWithData(data as CFData, nil) else { return nil }
        return CGImageSourceCreateImageAtIndex(source, 0, nil)
    }

    /// Preprocess image for VAE encoding.
    /// When no resize is needed, reads pixels directly from CGImage.dataProvider to
    /// preserve exact pixel values without format conversion or anti-aliasing artifacts.
    /// When a resize is needed, uses CGContext with high-quality interpolation, then
    /// reads from the resized image's dataProvider.
    private func preprocessImageForVAE(_ image: CGImage, targetHeight: Int, targetWidth: Int) -> MLXArray {
        let sourceWidth = image.width
        let sourceHeight = image.height

        // Resize image using CoreGraphics if needed
        let sourceImage: CGImage
        if sourceWidth != targetWidth || sourceHeight != targetHeight {
            Flux2Debug.log("Resizing image from \(sourceWidth)x\(sourceHeight) to \(targetWidth)x\(targetHeight)")

            let bytesPerPixel = 4
            let bytesPerRow = bytesPerPixel * targetWidth
            var pixelData = [UInt8](repeating: 0, count: targetHeight * bytesPerRow)

            guard let context = CGContext(
                data: &pixelData,
                width: targetWidth,
                height: targetHeight,
                bitsPerComponent: 8,
                bytesPerRow: bytesPerRow,
                space: CGColorSpaceCreateDeviceRGB(),
                bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue
            ) else {
                Flux2Debug.log("Failed to create resize context")
                return MLXRandom.normal([1, 3, targetHeight, targetWidth])
            }

            // High quality interpolation
            context.interpolationQuality = .high
            context.draw(image, in: CGRect(x: 0, y: 0, width: targetWidth, height: targetHeight))

            guard let resized = context.makeImage() else {
                Flux2Debug.log("Failed to create resized image")
                return MLXRandom.normal([1, 3, targetHeight, targetWidth])
            }

            sourceImage = resized
        } else {
            sourceImage = image
        }

        // Read pixels directly from CGImage data provider (no CGContext re-rendering).
        // This preserves exact pixel values without format conversion or anti-aliasing.
        let width = sourceImage.width
        let height = sourceImage.height
        let bpp = sourceImage.bitsPerPixel / 8  // bytes per pixel (3 for RGB, 4 for RGBA)
        let bpr = sourceImage.bytesPerRow

        guard let dataProvider = sourceImage.dataProvider,
              let pixelCFData = dataProvider.data,
              CFDataGetLength(pixelCFData) >= height * bpr else {
            // Fallback to CGContext if direct access fails (e.g. compressed or unusual format)
            Flux2Debug.log("Direct pixel access failed, falling back to CGContext")
            return preprocessImageForVAEViaCGContext(sourceImage, height: height, width: width)
        }

        let bytes = CFDataGetBytePtr(pixelCFData)!

        // Determine channel layout from bitmap info
        let alphaInfo = CGImageAlphaInfo(rawValue: sourceImage.bitmapInfo.rawValue & CGBitmapInfo.alphaInfoMask.rawValue)
        let hasAlpha = bpp >= 4
        let alphaFirst = alphaInfo == .premultipliedFirst || alphaInfo == .first || alphaInfo == .noneSkipFirst

        // RGB offsets depend on alpha position
        let rOffset: Int
        let gOffset: Int
        let bOffset: Int
        if hasAlpha && alphaFirst {
            // ARGB layout
            rOffset = 1; gOffset = 2; bOffset = 3
        } else {
            // RGB or RGBA layout
            rOffset = 0; gOffset = 1; bOffset = 2
        }

        // Convert to float array and normalize to [-1, 1]
        var floatData = [Float](repeating: 0, count: height * width * 3)
        for y in 0..<height {
            let rowStart = y * bpr
            for x in 0..<width {
                let pixelStart = rowStart + x * bpp
                let pixelIndex = y * width + x

                var r = Float(bytes[pixelStart + rOffset])
                var g = Float(bytes[pixelStart + gOffset])
                var b = Float(bytes[pixelStart + bOffset])

                // Un-premultiply alpha if needed
                if hasAlpha && (alphaInfo == .premultipliedFirst || alphaInfo == .premultipliedLast) {
                    let a = Float(bytes[pixelStart + (alphaFirst ? 0 : 3)])
                    if a > 0 && a < 255 {
                        let scale = 255.0 / a
                        r = min(r * scale, 255.0)
                        g = min(g * scale, 255.0)
                        b = min(b * scale, 255.0)
                    }
                }

                // Normalize to [-1, 1]
                floatData[pixelIndex] = r / 127.5 - 1.0
                floatData[height * width + pixelIndex] = g / 127.5 - 1.0
                floatData[2 * height * width + pixelIndex] = b / 127.5 - 1.0
            }
        }

        return MLXArray(floatData).reshaped([1, 3, height, width])
    }

    /// Fallback: read pixels via CGContext when direct data provider access is not possible.
    private func preprocessImageForVAEViaCGContext(_ image: CGImage, height: Int, width: Int) -> MLXArray {
        let bytesPerPixel = 4
        let bytesPerRow = bytesPerPixel * width
        var pixelData = [UInt8](repeating: 0, count: height * bytesPerRow)

        guard let context = CGContext(
            data: &pixelData,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: bytesPerRow,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue
        ) else {
            return MLXRandom.normal([1, 3, height, width])
        }

        context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))

        var floatData = [Float](repeating: 0, count: height * width * 3)
        for y in 0..<height {
            for x in 0..<width {
                let pixelIndex = y * width + x
                let byteIndex = y * bytesPerRow + x * bytesPerPixel

                floatData[pixelIndex] = Float(pixelData[byteIndex]) / 127.5 - 1.0
                floatData[height * width + pixelIndex] = Float(pixelData[byteIndex + 1]) / 127.5 - 1.0
                floatData[2 * height * width + pixelIndex] = Float(pixelData[byteIndex + 2]) / 127.5 - 1.0
            }
        }

        return MLXArray(floatData).reshaped([1, 3, height, width])
    }

    /// Convert VAE output to CGImage
    /// OPTIMIZED: Uses bulk array extraction instead of per-pixel loop
    private func postprocessVAEOutput(_ tensor: MLXArray) -> CGImage? {
        // tensor shape: [1, 3, H, W]
        let shape = tensor.shape
        guard shape.count == 4, shape[1] == 3 else {
            Flux2Debug.log("Unexpected tensor shape: \(shape)")
            return nil
        }

        let height = shape[2]
        let width = shape[3]

        // Denormalize from [-1, 1] to [0, 255] and convert to UInt8 in MLX
        // This does the conversion on GPU, much faster than CPU loop
        let denormalized = (tensor + 1.0) * 127.5
        let clamped = clip(denormalized, min: 0, max: 255)

        // Convert to [H, W, 3] layout for CGImage and cast to UInt8 on GPU
        let hwc = clamped.squeezed(axis: 0)  // [3, H, W]
            .transposed(axes: [1, 2, 0])      // [H, W, 3]
            .asType(.uint8)                    // Convert to UInt8 on GPU

        // Single eval and bulk extraction - MUCH faster than per-pixel loop
        eval(hwc)
        let pixelData = hwc.asArray(UInt8.self)

        // Create CGImage
        guard let providerRef = CGDataProvider(data: Data(pixelData) as CFData) else {
            return nil
        }

        return CGImage(
            width: width,
            height: height,
            bitsPerComponent: 8,
            bitsPerPixel: 24,
            bytesPerRow: width * 3,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.none.rawValue),
            provider: providerRef,
            decode: nil,
            shouldInterpolate: true,
            intent: .defaultIntent
        )
    }
}

// MARK: - Memory Management

extension Flux2Pipeline {
    /// Estimate memory requirement for current configuration
    public var estimatedMemoryGB: Int {
        quantization.estimatedTotalMemoryGB
    }

    /// Clear all loaded models and free memory
    public func clearAll() async {
        await unloadTextEncoder()
        unloadTransformer()
        vae = nil
        isLoaded = false
        memoryManager.fullCleanup()
    }
}

// MARK: - Model Status

extension Flux2Pipeline {
    /// Check if required models are downloaded
    public var hasRequiredModels: Bool {
        // Check transformer - use the appropriate variant for this model type and quantization
        let transformerVariant = ModelRegistry.TransformerVariant.variant(for: model, quantization: quantization.transformer)
        let hasTransformer = Flux2ModelDownloader.isDownloaded(.transformer(transformerVariant))

        // Text encoder is handled by MistralCore/FluxTextEncoders, skip check here

        // Check VAE
        let hasVAE = Flux2ModelDownloader.isDownloaded(.vae(vaeVariant))

        return hasTransformer && hasVAE
    }

    // =====================================================================
    // K4D LOCAL PATCH — strength-controlled img2img public entry point
    // (2026-05-15). Wraps `generate(mode: .strengthImageToImage(...))`
    // with a docstring + signature familiar to callers of the existing
    // `generateImageToImage` overloads. Delete this method to roll back.
    // =====================================================================

    /// Classical strength-controlled img2img — start from an init
    /// image at the requested denoising strength and run only the
    /// tail of the trajectory.
    ///
    /// Unlike `generateImageToImage(images:)` (which uses references
    /// as transformer-attention CONTEXT while still starting from
    /// pure noise), this method noises the init image's encoded
    /// latent and skips part of the denoise loop — the SD/Auto1111
    /// "denoising strength" pattern.
    ///
    /// Optionally accepts a non-empty `references` array (up to 3 for
    /// Klein). When refs are provided, the call routes to the
    /// `.strengthImageToImageWithRefs` mode which combines the
    /// init-encode + noise-blend latent setup with the I2I-style
    /// refs-as-attention-tokens conditioning — the user's init still
    /// drives the composition, the refs still drive context.
    ///
    /// - Parameters:
    ///   - prompt: Text prompt describing the desired output.
    ///   - initImage: Init image whose latent seeds the trajectory.
    ///   - strength: 0.0–1.0. Recommended 0.5–0.9 for Klein. 1.0
    ///     equals pure T2I (init contribution is mathematically 0).
    ///   - references: Optional Kontext refs (0–3). Default empty.
    ///   - height/width: Output dimensions. Default → init dims.
    ///   - steps: Denoising steps. Klein default 4.
    ///   - guidance: Distilled-CFG embed. Klein default 1.0.
    ///   - seed: Reproducible noise sample.
    ///   - upsamplePrompt: Visual prompt enhancement (default off).
    ///   - checkpointInterval: Save intermediate every N steps.
    /// - Returns: Generated CGImage.
    public func generateImageToImageWithStrength(
        prompt: String,
        initImage: CGImage,
        strength: Float,
        references: [CGImage] = [],
        height: Int? = nil,
        width: Int? = nil,
        steps: Int = 4,
        guidance: Float = 1.0,
        seed: UInt64? = nil,
        upsamplePrompt: Bool = false,
        checkpointInterval: Int? = nil,
        onProgress: Flux2ProgressCallback? = nil,
        onCheckpoint: Flux2CheckpointCallback? = nil
    ) async throws -> CGImage {
        let targetHeight = height ?? initImage.height
        let targetWidth = width ?? initImage.width

        // Route to the correct mode: refs-empty → existing strength
        // path (unchanged behavior, regression-safe); refs-present →
        // the new strength+refs handler.
        let mode: Flux2GenerationMode = references.isEmpty
            ? .strengthImageToImage(initImage: initImage, strength: strength)
            : .strengthImageToImageWithRefs(initImage: initImage, strength: strength, references: references)

        return try await generate(
            mode: mode,
            prompt: prompt,
            interpretImagePaths: nil,
            height: targetHeight,
            width: targetWidth,
            steps: steps,
            guidance: guidance,
            seed: seed,
            upsamplePrompt: upsamplePrompt,
            checkpointInterval: checkpointInterval,
            onProgress: onProgress,
            onCheckpoint: onCheckpoint
        )
    }

    // =====================================================================
    // END K4D LOCAL PATCH
    // =====================================================================

    /// List missing models
    public var missingModels: [ModelRegistry.ModelComponent] {
        var missing: [ModelRegistry.ModelComponent] = []

        // Use the appropriate transformer variant for this model type
        let transformerVariant = ModelRegistry.TransformerVariant.variant(for: model, quantization: quantization.transformer)
        if !Flux2ModelDownloader.isDownloaded(.transformer(transformerVariant)) {
            missing.append(.transformer(transformerVariant))
        }

        // Text encoder is handled by MistralCore/FluxTextEncoders

        if !Flux2ModelDownloader.isDownloaded(.vae(vaeVariant)) {
            missing.append(.vae(vaeVariant))
        }

        return missing
    }
}

