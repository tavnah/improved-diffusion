import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.font_manager as fm
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import numpy as np
from PIL import Image
from additional_scripts import main as mn
import cv2

LR = plt.imread('/data/GAN_project/test_imgs/onit_mit/2/COS7_tom20_647_unfiltered003-second_STORM_LR.tiff')
RECON = plt.imread('/data/GAN_project/test_imgs/onit_mit/2/COS7_tom20_647_unfiltered003-second_STORM_RECON.tiff')
HR = plt.imread('/data/GAN_project/mitochondria/onit/COS7_tom20_647_unfiltered003-second_STORM.jpg')

#LR_zoom = LR[180//4:310//4,260//4:485//4]
#RECON_zoom = RECON[180:310,260:485]
#HR_zoom = HR[180:310,260:485]
LR_ups = cv2.resize(LR, (1584, 816),  fx=0, fy=0, interpolation=cv2.INTER_NEAREST)
overlay = np.zeros((816, 1584, 3))
overlay[:,:,0] = mn.norm_percentile(RECON[:,:,0], 95)
overlay[:,:,1] = mn.norm_percentile(LR_ups[:,:,0], 95)
#overlay_zoom = overlay[180:310,260:485]
zoom_x, zoom_y = 175*4, 110*4
width, height = 75*4, 50*4

LR_zoom = LR[zoom_y//4:(zoom_y+height)//4,zoom_x//4:(zoom_x+width)//4]
HR_zoom = HR[zoom_y:(zoom_y+height),zoom_x:(zoom_x+width)]
RECON_zoom = RECON[zoom_y:(zoom_y+height),zoom_x:(zoom_x+width)]

# p1 = plt.subplot(131)
# plt.subplot(132, share=p1)
# plt.subplot(133, share=p1)

fig2 = plt.figure(constrained_layout=True, figsize=(20, 8))
gs = fig2.add_gridspec(2, 3)
ax1 = fig2.add_subplot(gs[0, 0])
ax1.imshow(LR[:,:,0], cmap='gray')
ax1.set_title('Simulated Widefield', fontsize=35)
fontprops = fm.FontProperties(size=20)
rect1 = patches.Rectangle((zoom_x//4,zoom_y//4), width//4, height//4, linewidth=3, edgecolor='w', facecolor='none',linestyle='--')
ax1.add_patch(rect1)
ax2 = fig2.add_subplot(gs[0, 1])
ax2.imshow(RECON[:,:,0], cmap='gray')
ax2.set_title('Care Reconstruction',fontsize=35)
rect2 = patches.Rectangle((zoom_x, zoom_y), width, height, linewidth=3, edgecolor='w', facecolor='none',linestyle='--')
ax2.add_patch(rect2)
ax3 = fig2.add_subplot(gs[0, 2])
ax3.imshow(HR, cmap='gray')
ax3.set_title('Ground Truth',fontsize=35)
rect3 = patches.Rectangle((zoom_x, zoom_y), width, height, linewidth=3, edgecolor='w', facecolor='none',linestyle='--')
ax3.add_patch(rect3)
scalebar3 = AnchoredSizeBar(ax3.transData,
                           31*4, '$5 \mu m$', 'lower left',
                           pad=0.1,
                           color='white',
                           frameon=False,
                           size_vertical=1,
                           fontproperties=fontprops)
ax3.add_artist(scalebar3)

ax5 = fig2.add_subplot(gs[1, 0])
ax5.imshow(LR_zoom[:,:,0], cmap='gray')
ax6 = fig2.add_subplot(gs[1, 1])
ax6.imshow(RECON_zoom[:,:,0], cmap='gray')
ax7 = fig2.add_subplot(gs[1, 2])
ax7.imshow(HR_zoom, cmap='gray')
scalebar7 = AnchoredSizeBar(ax7.transData,
                           6*4, '$1 \mu m$', 'lower left',
                           pad=0.1,
                           color='white',
                           frameon=False,
                           size_vertical=1,
                           fontproperties=fontprops)
ax7.add_artist(scalebar7)

plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
plt.show()

#image = Image.frombytes('RGB', fig.canvas.get_width_height(), fig.canvas.tostring_rgb())
#image.save('/data/GAN_project/test_imgs/shareloc_MT3D_160530_C1C2_758K/poster/comparison_fig_mito2.bmp')
