import SwiftUI
import UIKit

enum GlassHaptics {
    static func light() {
        let gen = UIImpactFeedbackGenerator(style: .light)
        gen.impactOccurred()
    }

    static func medium() {
        let gen = UIImpactFeedbackGenerator(style: .medium)
        gen.impactOccurred()
    }

    static func selection() {
        let gen = UISelectionFeedbackGenerator()
        gen.selectionChanged()
    }
}

/// Tactile spring scale + specular flare button style for Liquid Glass controls
struct LiquidGlassPressStyle: ButtonStyle {
    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .scaleEffect(configuration.isPressed ? 0.94 : 1.0)
            .brightness(configuration.isPressed ? 0.07 : 0.0)
            .animation(.spring(response: 0.26, dampingFraction: 0.68), value: configuration.isPressed)
    }
}

/// Multi-layered Apple Liquid Glass Capsule / Rounded-Rect Surface Modifier
struct LiquidGlassCapsule: ViewModifier {
    @Environment(\.colorScheme) private var colorScheme
    var cornerRadius: CGFloat = 28
    var isActive: Bool = false
    var tintColor: Color = Color(red: 0.30, green: 0.80, blue: 1.0)

    func body(content: Content) -> some View {
        let isLight = (colorScheme == .light)
        let shape = RoundedRectangle(cornerRadius: cornerRadius, style: .continuous)
        let glazeColors: [Color] = isLight
            ? (isActive
                ? [tintColor.opacity(0.28), tintColor.opacity(0.14), Color.white.opacity(0.65)]
                : [Color.white.opacity(0.68), Color.white.opacity(0.32), tintColor.opacity(0.08)])
            : (isActive
                ? [tintColor.opacity(0.38), tintColor.opacity(0.16), Color.white.opacity(0.10)]
                : [Color.white.opacity(0.13), Color.white.opacity(0.03), tintColor.opacity(0.05)])
        let borderTop = isLight
            ? Color.white.opacity(isActive ? 0.96 : 0.85)
            : Color.white.opacity(isActive ? 0.78 : 0.52)
        let borderMid = isLight ? Color.black.opacity(0.06) : Color.white.opacity(0.12)
        let borderBot = tintColor.opacity(isActive ? 0.72 : (isLight ? 0.24 : 0.28))

        content
            .background(
                ZStack {
                    shape.fill(.ultraThinMaterial)

                    // Inner refractive glaze
                    shape.fill(
                        LinearGradient(
                            colors: glazeColors,
                            startPoint: .topLeading,
                            endPoint: .bottomTrailing
                        )
                    )
                }
            )
            .overlay(
                // Dual-angle specular refraction rim
                shape.strokeBorder(
                    LinearGradient(
                        stops: [
                            .init(color: borderTop, location: 0.0),
                            .init(color: borderMid, location: 0.42),
                            .init(color: borderBot, location: 1.0)
                        ],
                        startPoint: .topLeading,
                        endPoint: .bottomTrailing
                    ),
                    lineWidth: isActive ? 1.25 : 0.9
                )
            )
            .shadow(
                color: isActive ? tintColor.opacity(isLight ? 0.22 : 0.28) : Color.black.opacity(isLight ? 0.12 : 0.34),
                radius: isActive ? 14 : 10,
                x: 0,
                y: isLight ? 4 : 6
            )
    }
}

extension View {
    func liquidGlass(cornerRadius: CGFloat = 28, isActive: Bool = false, tint: Color = Color(red: 0.30, green: 0.80, blue: 1.0)) -> some View {
        modifier(LiquidGlassCapsule(cornerRadius: cornerRadius, isActive: isActive, tintColor: tint))
    }
}

/// Circular Liquid Glass Orb Icon Button
struct LiquidGlassOrbButton: View {
    @Environment(\.colorScheme) private var colorScheme
    let systemImage: String
    var size: CGFloat = 42
    var isActive: Bool = false
    var tint: Color = Color(red: 0.30, green: 0.80, blue: 1.0)
    let action: () -> Void

    var body: some View {
        let isLight = (colorScheme == .light)
        let fg: Color = isLight
            ? (isActive ? Color(red: 0.04, green: 0.28, blue: 0.62) : Color(red: 0.09, green: 0.15, blue: 0.25))
            : (isActive ? Color.white : Color.white.opacity(0.92))
        Button {
            GlassHaptics.light()
            action()
        } label: {
            Image(systemName: systemImage)
                .font(.system(size: size * 0.40, weight: .semibold))
                .foregroundStyle(fg)
                .frame(width: size, height: size)
                .liquidGlass(cornerRadius: size * 0.5, isActive: isActive, tint: tint)
        }
        .buttonStyle(LiquidGlassPressStyle())
    }
}

/// Sliding Refractive Lens Segmented Control
struct LiquidGlassSegmentedControl: View {
    @Environment(\.colorScheme) private var colorScheme
    let items: [(icon: String?, title: String)]
    @Binding var selection: Int
    @Namespace private var lensNamespace

    var body: some View {
        let isLight = (colorScheme == .light)
        let activeFg: Color = isLight ? Color(red: 0.04, green: 0.28, blue: 0.62) : .white
        let inactiveFg: Color = isLight ? Color(red: 0.28, green: 0.36, blue: 0.49) : Color.white.opacity(0.68)
        let lensColors: [Color] = isLight
            ? [Color.white.opacity(0.92), Color(red: 0.22, green: 0.72, blue: 1.0).opacity(0.24)]
            : [Color.white.opacity(0.28), Color(red: 0.25, green: 0.78, blue: 1.0).opacity(0.28)]
        let lensBorder: [Color] = isLight
            ? [Color.white, Color(red: 0.12, green: 0.56, blue: 0.94).opacity(0.35)]
            : [Color.white.opacity(0.75), Color.white.opacity(0.15)]

        HStack(spacing: 4) {
            ForEach(items.indices, id: \.self) { idx in
                let active = (selection == idx)
                Button {
                    guard selection != idx else { return }
                    GlassHaptics.selection()
                    withAnimation(.spring(response: 0.32, dampingFraction: 0.76)) {
                        selection = idx
                    }
                } label: {
                    HStack(spacing: 5) {
                        if let ic = items[idx].icon {
                            Image(systemName: ic)
                                .font(.system(size: 13, weight: .semibold))
                        }
                        Text(items[idx].title)
                            .font(.system(size: 13, weight: .semibold, design: .rounded))
                            .lineLimit(1)
                    }
                    .foregroundStyle(active ? activeFg : inactiveFg)
                    .frame(maxWidth: .infinity)
                    .frame(height: 40)
                    .background {
                        if active {
                            Capsule(style: .continuous)
                                .fill(
                                    LinearGradient(
                                        colors: lensColors,
                                        startPoint: .topLeading,
                                        endPoint: .bottomTrailing
                                    )
                                )
                                .overlay(
                                    Capsule(style: .continuous)
                                        .strokeBorder(
                                            LinearGradient(
                                                colors: lensBorder,
                                                startPoint: .top,
                                                endPoint: .bottom
                                            ),
                                            lineWidth: 1.0
                                        )
                                )
                                .shadow(color: Color.cyan.opacity(isLight ? 0.16 : 0.22), radius: 8, x: 0, y: 2)
                                .matchedGeometryEffect(id: "glass_lens", in: lensNamespace)
                        }
                    }
                }
                .buttonStyle(.plain)
            }
        }
        .padding(5)
        .liquidGlass(cornerRadius: 26)
    }
}

/// Liquid Glass Choice Pill
struct LiquidGlassChip: View {
    @Environment(\.colorScheme) private var colorScheme
    let title: String
    let isSelected: Bool
    let action: () -> Void

    var body: some View {
        let isLight = (colorScheme == .light)
        let fg: Color = isLight
            ? (isSelected ? Color(red: 0.04, green: 0.28, blue: 0.62) : Color(red: 0.14, green: 0.21, blue: 0.32))
            : (isSelected ? Color.white : Color.white.opacity(0.82))
        Button {
            GlassHaptics.selection()
            action()
        } label: {
            HStack(spacing: 5) {
                if isSelected {
                    Image(systemName: "checkmark")
                        .font(.system(size: 11, weight: .bold))
                }
                Text(title)
                    .font(.system(size: 13, weight: .semibold, design: .rounded))
                    .lineLimit(1)
            }
            .foregroundStyle(fg)
            .padding(.horizontal, isSelected ? 13 : 15)
            .frame(height: 36)
            .liquidGlass(cornerRadius: 18, isActive: isSelected)
        }
        .buttonStyle(LiquidGlassPressStyle())
    }
}

/// Optical Liquid Droplet Swatch Pill
struct LiquidDropletSwatch: View {
    @Environment(\.colorScheme) private var colorScheme
    let preset: MaterialPreset
    let isSelected: Bool
    let action: () -> Void

    var body: some View {
        let isLight = (colorScheme == .light)
        let fg: Color = isLight
            ? (isSelected ? Color(red: 0.04, green: 0.26, blue: 0.58) : Color(red: 0.12, green: 0.19, blue: 0.30))
            : (isSelected ? Color.white : Color.white.opacity(0.85))
        Button {
            GlassHaptics.selection()
            action()
        } label: {
            HStack(spacing: 8) {
                ZStack {
                    Circle()
                        .fill(
                            RadialGradient(
                                colors: [preset.shallowColor, preset.deepColor],
                                center: .topLeading,
                                startRadius: 2,
                                endRadius: 18
                            )
                        )
                        .frame(width: 20, height: 20)
                    Circle()
                        .fill(Color.white.opacity(0.78))
                        .frame(width: 5, height: 5)
                        .offset(x: -3.5, y: -3.5)
                    Circle()
                        .strokeBorder(isLight ? Color.black.opacity(0.18) : Color.white.opacity(0.45), lineWidth: 0.8)
                        .frame(width: 20, height: 20)
                }

                Text(preset.name)
                    .font(.system(size: 13, weight: .semibold, design: .rounded))
                    .foregroundStyle(fg)
                    .lineLimit(1)
            }
            .padding(.leading, 10)
            .padding(.trailing, 14)
            .frame(height: 40)
            .liquidGlass(cornerRadius: 20, isActive: isSelected, tint: preset.shallowColor)
        }
        .buttonStyle(LiquidGlassPressStyle())
    }
}

/// Frosted Glass Card Container inside the Liquid Glass Sheet
struct LiquidGlassCard<Content: View>: View {
    @Environment(\.colorScheme) private var colorScheme
    let title: String?
    @ViewBuilder let content: Content

    init(_ title: String? = nil, @ViewBuilder content: () -> Content) {
        self.title = title
        self.content = content()
    }

    var body: some View {
        let isLight = (colorScheme == .light)
        let cardFill = isLight ? Color.white.opacity(0.56) : Color.white.opacity(0.065)
        let borderColors = isLight
            ? [Color.white.opacity(0.95), Color.black.opacity(0.08)]
            : [Color.white.opacity(0.32), Color.white.opacity(0.06)]
        let titleColor = isLight ? Color(red: 0.08, green: 0.14, blue: 0.24) : Color.white.opacity(0.92)

        VStack(alignment: .leading, spacing: 12) {
            if let title = title {
                Text(title)
                    .font(.system(size: 14, weight: .semibold, design: .rounded))
                    .foregroundStyle(titleColor)
            }
            content
        }
        .padding(16)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(
            RoundedRectangle(cornerRadius: 24, style: .continuous)
                .fill(cardFill)
                .overlay(
                    RoundedRectangle(cornerRadius: 24, style: .continuous)
                        .strokeBorder(
                            LinearGradient(
                                colors: borderColors,
                                startPoint: .topLeading,
                                endPoint: .bottomTrailing
                            ),
                            lineWidth: 0.85
                        )
                )
        )
    }
}

/// Clean Liquid Glass Parameter Slider Row
struct LiquidGlassSlider: View {
    @Environment(\.colorScheme) private var colorScheme
    let label: String
    @Binding var value: Float
    let range: ClosedRange<Float>
    var format: String = "%.2f"

    var body: some View {
        let isLight = (colorScheme == .light)
        let labelCol = isLight ? Color(red: 0.14, green: 0.22, blue: 0.34) : Color.white.opacity(0.84)
        let pillFg = isLight ? Color(red: 0.04, green: 0.36, blue: 0.72) : Color(red: 0.65, green: 0.90, blue: 1.0)
        let pillBg = isLight ? Color.black.opacity(0.06) : Color.white.opacity(0.10)
        let tintCol = isLight ? Color(red: 0.05, green: 0.45, blue: 0.86) : Color(red: 0.36, green: 0.82, blue: 1.0)

        VStack(spacing: 5) {
            HStack {
                Text(label)
                    .font(.system(size: 13, weight: .medium, design: .rounded))
                    .foregroundStyle(labelCol)
                Spacer()
                Text(String(format: format, value))
                    .font(.system(size: 12, weight: .semibold, design: .monospaced))
                    .foregroundStyle(pillFg)
                    .padding(.horizontal, 8)
                    .padding(.vertical, 2)
                    .background(Capsule().fill(pillBg))
            }
            Slider(
                value: Binding(
                    get: { Double(value) },
                    set: { value = Float($0) }
                ),
                in: Double(range.lowerBound)...Double(range.upperBound)
            )
            .tint(tintCol)
        }
    }
}
