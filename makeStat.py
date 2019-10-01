import cv2
import glob
from tqdm.autonotebook import tqdm
from skimage.measure import label, regionprops

class_names = ["Stroma", "Tumor", "Immune_cells", "Other"]
files = glob.glob('/home/maorvelous/Documents/Lab/deepLearning/MaskLou/*.png')
bases = list(set(["_".join(f.split('_')[0:2]).replace("./masks\\", "") for f in files]))

bases

outfile = open("output.tsv", 'w')
for base in tqdm(bases):  # now for each of the files
    outfile.write(f"{base}\t")
    for class_name in tqdm(class_names):
        try:
            fname = f"{base}_{class_name}_mask.png"
            print(fname)
            im = cv2.imread(fname)
            label_im = label(im)
            props = regionprops(label_im)

            nitems = len(props)
            area = sum([p.area for p in props])
            avg = area / nitems
            outfile.write(f"{nitems}\t{area}\t{avg}")

        except:
            outfile.write(f"0\t0\t0")
            continue
    outfile.write("\n")
outfile.close()
