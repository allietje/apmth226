# convert_png_to_pdf.py
"""
convert_png_to_pdf.py - Convert specific PNG files to PDF format
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
    """Convert the 4 specified PNG files to PDF"""
    
    # Base directory
    base_dir = "../final_results/datasize"
    
    # Files to convert
    files_to_convert = [
        "final_error_vs_samples.png",
        "convergence_speed_vs_samples.png", 
        "curves_size1000.png",
        "curves_size5000.png",
        "curves_size30000.png",
        "curves_size60000.png"
    ]
    
    print("Converting PNG files to PDF format...")
    
    for filename in files_to_convert:
        png_path = os.path.join(base_dir, filename)
        pdf_filename = filename.replace('.png', '.pdf')
        pdf_path = os.path.join(base_dir, pdf_filename)
        
        if os.path.exists(png_path):
            success = convert_png_to_pdf(png_path, pdf_path)
            if success:
                print(f"✓ {filename} -> {pdf_filename}")
        else:
            print(f"✗ {filename} not found")

if __name__ == "__main__":
    main()