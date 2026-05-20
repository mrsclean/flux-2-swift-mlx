// Flux2ChainsTests.swift — Unit tests for chain helpers
// Copyright 2025 Vincent Gourbin

import XCTest
@testable import Flux2Chains
import Flux2Core
import CoreGraphics

final class Flux2ChainHelpersTests: XCTestCase {

    // MARK: - Dimension resolution

    func testResolveDimensionsLeavesValidInputUntouched() {
        let (h, w) = Flux2Pipeline.resolveChainDimensions(width: 512, height: 512)
        XCTAssertEqual(h, 512)
        XCTAssertEqual(w, 512)
    }

    func testResolveDimensionsClampsToMultipleOf32() {
        let (h, w) = Flux2Pipeline.resolveChainDimensions(width: 1000, height: 600)
        XCTAssertEqual(w % 32, 0)
        XCTAssertEqual(h % 32, 0)
        XCTAssertLessThanOrEqual(w, 1000)
        XCTAssertLessThanOrEqual(h, 600)
    }

    func testResolveDimensionsScalesDownLargeInputs() {
        // 4K input should be scaled into ≤ 1024² area, multiple of 32.
        let (h, w) = Flux2Pipeline.resolveChainDimensions(width: 3840, height: 2160)
        XCTAssertLessThanOrEqual(w * h, 1024 * 1024)
        XCTAssertEqual(w % 32, 0)
        XCTAssertEqual(h % 32, 0)
        // Aspect preserved within 5% (rounding allowed).
        let inputAspect = 3840.0 / 2160.0
        let outputAspect = Double(w) / Double(h)
        XCTAssertEqual(inputAspect, outputAspect, accuracy: 0.1)
    }

    func testResolveDimensionsRespectsCustomMaxPixels() {
        let (h, w) = Flux2Pipeline.resolveChainDimensions(width: 1024, height: 1024, maxPixels: 512 * 512)
        XCTAssertLessThanOrEqual(w * h, 512 * 512)
        XCTAssertEqual(w % 32, 0)
        XCTAssertEqual(h % 32, 0)
    }

    // MARK: - Mask packing

    func testPackMaskHasExpectedSequenceShape() {
        // 512×512 image → 32×32 latent grid → 1024 tokens.
        let mask = makeUniformMask(width: 512, height: 512, value: 1.0)
        let packed = Flux2Pipeline.packMaskForLatentBlending(mask, targetHeight: 512, targetWidth: 512)
        XCTAssertEqual(packed.shape, [1, 1024, 1])
    }

    func testPackMaskAllWhiteYieldsOnes() {
        let mask = makeUniformMask(width: 128, height: 128, value: 1.0)
        let packed = Flux2Pipeline.packMaskForLatentBlending(mask, targetHeight: 128, targetWidth: 128)
        // 128/16 = 8 → 64 tokens
        XCTAssertEqual(packed.shape, [1, 64, 1])
        // Sum should equal token count.
        let total = packed.sum().item(Float.self)
        XCTAssertEqual(total, 64.0, accuracy: 0.5)
    }

    func testPackMaskAllBlackYieldsZeros() {
        let mask = makeUniformMask(width: 128, height: 128, value: 0.0)
        let packed = Flux2Pipeline.packMaskForLatentBlending(mask, targetHeight: 128, targetWidth: 128)
        let total = packed.sum().item(Float.self)
        XCTAssertEqual(total, 0.0, accuracy: 0.5)
    }

    // MARK: - Helpers

    private func makeUniformMask(width: Int, height: Int, value: Float) -> CGImage {
        let pixelValue: UInt8 = UInt8(max(0, min(255, Int(value * 255))))
        let pixelCount = width * height
        var pixels = [UInt8](repeating: pixelValue, count: pixelCount)
        let colorSpace = CGColorSpaceCreateDeviceGray()
        let context = CGContext(
            data: &pixels,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: width,
            space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.none.rawValue
        )!
        return context.makeImage()!
    }
}
