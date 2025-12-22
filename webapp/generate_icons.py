"""
Icon Generator for PWA
This script generates all required icon sizes for your Progressive Web App.
You need to have a source image (e.g., logo.png) at least 512x512 pixels.
"""

from PIL import Image, ImageDraw, ImageFont
import os

def create_simple_icon(size, output_path):
    """Create a simple medical-themed icon"""
    # Create a new image with the theme color
    img = Image.new('RGB', (size, size), '#a5b68d')
    draw = ImageDraw.Draw(img)

    # Draw a white circle (representing medical/health theme)
    padding = size // 6
    circle_bbox = [padding, padding, size - padding, size - padding]
    draw.ellipse(circle_bbox, fill='white', outline='#a5b68d', width=size//40)

    # Draw a cross symbol (medical symbol)
    cross_width = size // 10
    cross_height = size // 3
    center_x = size // 2
    center_y = size // 2

    # Vertical bar of cross
    draw.rectangle([
        center_x - cross_width // 2,
        center_y - cross_height // 2,
        center_x + cross_width // 2,
        center_y + cross_height // 2
    ], fill='#a5b68d')

    # Horizontal bar of cross
    draw.rectangle([
        center_x - cross_height // 2,
        center_y - cross_width // 2,
        center_x + cross_height // 2,
        center_y + cross_width // 2
    ], fill='#a5b68d')

    # Save the icon
    img.save(output_path, 'PNG')
    print(f"✓ Created {size}x{size} icon at {output_path}")

def generate_all_icons():
    """Generate all required icon sizes"""
    # Get the script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    icons_dir = os.path.join(script_dir, 'static', 'icons')

    # Create icons directory if it doesn't exist
    os.makedirs(icons_dir, exist_ok=True)

    # Icon sizes required for PWA
    sizes = [72, 96, 128, 144, 152, 192, 384, 512]

    print("Generating PWA icons...")
    print("=" * 50)

    for size in sizes:
        output_path = os.path.join(icons_dir, f'icon-{size}x{size}.png')
        create_simple_icon(size, output_path)

    print("=" * 50)
    print("✓ All icons generated successfully!")
    print(f"\nIcons saved to: {icons_dir}")
    print("\nNote: You can replace these auto-generated icons with your own")
    print("custom logo by placing images with the same names in the icons folder.")

if __name__ == '__main__':
    try:
        generate_all_icons()
    except Exception as e:
        print(f"Error generating icons: {e}")
        print("\nMake sure you have Pillow installed:")
        print("pip install Pillow")
