"""
convert_grad_norms.py - Convert grad_norms.png to PDF format
"""

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

def convert_png_to_pdf(png_path, pdf_path):
    """Convert a PNG file to PDF format"""
    try:
        # Read the PNG image
        img = mpimg.imread(png_path)
        
        # Create figure with the same aspect ratio
        height, width = img.shape[:2]
        aspect_ratio = width / height
        
        fig, ax = plt.subplots(figsize=(8, 8/aspect_ratio))
        ax.imshow(img)
        ax.axis('off')
        
        # Save as PDF
        fig.savefig(pdf_path, format='pdf', dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        print(f"Converted: {png_path} -> {pdf_path}")
        return True
        
    except Exception as e:
        print(f"Error converting {png_path}: {e}")
        return False

def main():
    """Convert the grad_norms.png file to PDF"""
    
    # Path to the grad_norms.png file
    png_path = "../final_results/30x800/grad_norms.png"
    pdf_path = "../final_results/30x800/grad_norms.pdf"
    
    if os.path.exists(png_path):
        success = convert_png_to_pdf(png_path, pdf_path)
        if success:
            print("✓ Successfully converted grad_norms.png to PDF")
        else:
            print("✗ Failed to convert the file")
    else:
        print(f"✗ File not found: {png_path}")

if __name__ == "__main__":
    main()