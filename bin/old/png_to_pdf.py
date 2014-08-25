from glob import glob
import os.path

from PyPDF2 import PdfFileMerger
from PIL import Image, ImageDraw

CURDIR = os.path.abspath('../res/ALL_GOOD')

def png_to_pdf(curdir):
    for infile in glob(os.path.join(curdir, '*.png')):
        infile = os.path.abspath(infile)
        outdir = os.path.join(os.path.dirname(infile), 'pdf')
        if not os.path.exists(outdir):
            os.mkdir(outdir)
        basefilename = os.path.splitext(os.path.split(infile)[-1])[0]
        outfile = basefilename + '.pdf'
        outfile = os.path.join(outdir, outfile)
        im = Image.open(infile)
        im = im.convert('RGB')

        draw = ImageDraw.Draw(im)
        title = basefilename
        draw.text((10,10), title, fill=(0,0,0))

        im.save(outfile, "PDF", Quality=100)

def merge_pdfs(curdir, outfile):
    merger = PdfFileMerger()
    for infile in glob(os.path.join(curdir, '*.pdf')):
        merger.append(open(infile, 'rb'))
    merger.write(outfile)
    print outfile

if __name__ == "__main__":
    png_to_pdf(CURDIR)
    # outfile = os.path.abspath(os.path.join(CURDIR, 'all.pdf'))
    # merge_pdfs(CURDIR, outfile)
