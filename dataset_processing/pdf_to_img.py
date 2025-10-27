from pathlib import Path
from typing import List, Union
import fitz

"""
pdf_to_img.py

Convert every page of every PDF in a directory into image files saved in an output directory.
Requires: PyMuPDF
"""


def convert_pdfs_to_images(
    src_dir: Union[str, Path],
    dst_dir: Union[str, Path],
    fmt: str = "png",
    dpi: int = 200,
    recursive: bool = False,
    create_subdir_per_pdf: bool = False,
) -> List[Path]:
    """
    Walk through src_dir, convert each PDF page to an image, and save in dst_dir.
    Returns a list of saved image Path objects.

    Parameters:
    - src_dir: directory containing PDFs
    - dst_dir: directory to save images
    - fmt: image format/extension (e.g. "png", "jpg")
    - dpi: target resolution in DPI
    - recursive: if True, traverse src_dir recursively
    - create_subdir_per_pdf: if True, put pages of each PDF into a subfolder named after the PDF (default True)
    """
    src = Path(src_dir)
    dst = Path(dst_dir)
    if not src.is_dir():
        raise ValueError(f"Source directory does not exist: {src}")
    dst.mkdir(parents=True, exist_ok=True)

    pattern = "**/*.pdf" if recursive else "*.pdf"
    saved_files: List[Path] = []
    scale = dpi / 72  # PDF points are 72 DPI; scale matrix applied to render resolution
    mat = fitz.Matrix(scale, scale)

    for pdf_path in src.glob(pattern):
        if not pdf_path.is_file():
            continue
        try:
            doc = fitz.open(pdf_path)
        except Exception as e:
            print(f"Skipping '{pdf_path}': cannot open ({e})")
            continue

        if doc.is_encrypted:
            print(f"Skipping encrypted PDF without password: {pdf_path}")
            doc.close()
            continue

        base_out = dst / pdf_path.stem if create_subdir_per_pdf else dst
        base_out.mkdir(parents=True, exist_ok=True)      
    
        for i, page in enumerate(doc, start=1):
            try:
                pix = page.get_pixmap(matrix=mat, alpha=False)
                out_name = f"{pdf_path.stem}_{i:03d}.{fmt}"
                out_path = base_out / out_name
                pix.save(str(out_path))
                saved_files.append(out_path)
            except Exception as e:
                print(f"Failed to render page {i} of '{pdf_path}': {e}")

        doc.close()

    return saved_files


if __name__ == "__main__":
    # Example usage:
    results = convert_pdfs_to_images("datasets/snps/docs", "datasets/snps/imgs", fmt="png", dpi=300, recursive=True)
    print(f"Saved {len(results)} images.")