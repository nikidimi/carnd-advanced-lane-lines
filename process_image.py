import sys
import matplotlib.image as mpimg
from process_video import VideoLineDrawer

  
if __name__ == "__main__":
    ld = VideoLineDrawer()
    image = mpimg.imread(sys.argv[1])
    processed_image = ld.plot_image(image)
    mpimg.imsave("out.jpg", processed_image )
