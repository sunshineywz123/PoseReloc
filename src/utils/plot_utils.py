import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import random
import cv2
matplotlib.use('Agg')

def plot_image_pair(imgs, dpi=100, size=6, pad=.5):
    n = len(imgs)
    assert n == 2, 'number of images must be two'
    figsize = (size*n, size*3/4) if size is not None else None
    _, ax = plt.subplots(1, n, figsize=figsize, dpi=dpi)
    for i in range(n):
        ax[i].imshow(imgs[i], cmap=plt.get_cmap('gray'), vmin=0, vmax=255)
        ax[i].get_yaxis().set_ticks([])
        ax[i].get_xaxis().set_ticks([])
        for spine in ax[i].spines.values():  # remove frame
            spine.set_visible(False)
    plt.tight_layout(pad=pad)


def plot_keypoints(kpts0, kpts1, color='w', ps=2):
    ax = plt.gcf().axes
    ax[0].scatter(kpts0[:, 0], kpts0[:, 1], c=color, s=ps, marker='x')
    ax[1].scatter(kpts1[:, 0], kpts1[:, 1], c=color, s=ps, marker='x')

def plot_keypoints_for_img0(kpts,color='w',ps=2):
    ax = plt.gcf().axes
    ax[0].scatter(kpts[:, 0], kpts[:, 1], c=color, s=ps)

def plot_keypoints_for_img1(kpts,color='w',ps=2):
    ax = plt.gcf().axes
    ax[1].scatter(kpts[:, 0], kpts[:, 1], c=color, s=ps)

def plot_matches(kpts0, kpts1, color, lw=0.5, ps=4):
    fig = plt.gcf()
    ax = fig.axes
    fig.canvas.draw()

    transFigure = fig.transFigure.inverted()
    fkpts0 = transFigure.transform(ax[0].transData.transform(kpts0))
    fkpts1 = transFigure.transform(ax[1].transData.transform(kpts1))

    fig.lines = [matplotlib.lines.Line2D(
        (fkpts0[i, 0], fkpts1[i, 0]), (fkpts0[i, 1], fkpts1[i, 1]), zorder=1,
        transform=fig.transFigure, c=color[i], linewidth=lw)
        for i in range(len(kpts0))]
    ax[0].scatter(kpts0[:, 0], kpts0[:, 1], c=color, s=ps)
    ax[1].scatter(kpts1[:, 0], kpts1[:, 1], c=color, s=ps)

def plot_local_windows(kpts, color='r', lw=1, ax_=0, window_size=9):
    ax = plt.gcf().axes
    
    patches = []
    for kpt in kpts:
        #patches.append(matplotlib.patches.Rectangle((kpt[0],kpt[1]),window_size,window_size))
        ax[ax_].add_patch(matplotlib.patches.Rectangle((kpt[0]-(window_size//2)-1,kpt[1]-(window_size//2)-1), window_size+2, window_size+2, lw=lw,color=color, fill=False))
    #ax[ax_].add_collection(matplotlib.collections.PathCollection(patches))

def make_matching_plot(image0, image1, kpts0, kpts1, mkpts0, mkpts1,
                       color, text, path=None, show_keypoints=False,
                       fast_viz=False, opencv_display=False,
                       opencv_title='matches', small_text=[]):

    if fast_viz:
        make_matching_plot_fast(image0, image1, kpts0, kpts1, mkpts0, mkpts1,
                                color, text, path, show_keypoints, 10,
                                opencv_display, opencv_title, small_text)
        return

    plot_image_pair([image0, image1])  # will create a new figure
    if show_keypoints:
        plot_keypoints(kpts0, kpts1, color='k', ps=4)
        plot_keypoints(kpts0, kpts1, color='w', ps=2)
    plot_matches(mkpts0, mkpts1, color)

    fig = plt.gcf()
    txt_color = 'k' if image0[:100, :150].mean() > 200 else 'w'
    fig.text(
        0.01, 0.99, '\n'.join(text), transform=fig.axes[0].transAxes,
        fontsize=15, va='top', ha='left', color=txt_color)

    txt_color = 'k' if image0[-100:, :150].mean() > 200 else 'w'
    fig.text(
        0.01, 0.01, '\n'.join(small_text), transform=fig.axes[0].transAxes,
        fontsize=5, va='bottom', ha='left', color=txt_color)
    if path:
        plt.savefig(str(path), bbox_inches='tight', pad_inches=0)
        plt.close()
    else:
        # TODO: Would it leads to any issue without current figure opened?
        return fig

def make_matching_plot_show_more(image0, image1, kpts0, kpts1, mkpts0, mkpts1,
                       color, text, path=None, show_keypoints=False,draw_matches=True,
                       fast_viz=False, opencv_display=False,
                       opencv_title='matches', small_text=[],keypoints_more_0_a=None,keypoints_more_0_b=None,keypoints_more_1_a=None,
                       draw_local_window=None, window_size=9, mkpts_mask=None):

    if fast_viz:
        make_matching_plot_fast(image0, image1, kpts0, kpts1, mkpts0, mkpts1,
                                color, text, path, show_keypoints, 10,
                                opencv_display, opencv_title, small_text)
        return

    plot_image_pair([image0, image1],dpi=600)  # will create a new figure

    blue_points_size = 0.3  # means coarse full matches
    yellow_points_size = 0.1  # means detector's output keypoints
    green_points_size = 0.1 # means final matches

    print(f"coarse: {len(keypoints_more_0_a)}, spp keypoints:{len(keypoints_more_0_b)}, total matches:{len(mkpts0)}")
    if keypoints_more_0_a is not None:
        plot_keypoints_for_img0(keypoints_more_0_a,color='blue',ps=blue_points_size)
    if keypoints_more_0_b is not None:
        plot_keypoints_for_img0(keypoints_more_0_b,color='yellow',ps=yellow_points_size)
    if keypoints_more_1_a is not None:
        plot_keypoints_for_img1(keypoints_more_1_a,color='blue',ps=blue_points_size)
    if show_keypoints:
        plot_keypoints(kpts0, kpts1, color='green', ps=green_points_size)

    if draw_matches:
        mkpts0_ = mkpts0[mkpts_mask>0] if mkpts_mask is not None else mkpts0
        mkpts1_ = mkpts1[mkpts_mask>0] if mkpts_mask is not None else mkpts1
        n = len(mkpts0_)
        n_matches = 10000
        if n < n_matches:
            kpts_indices = np.arange(n)
        else:
            kpts_indices = random.sample(range(0,n),n_matches)
        mkpts0_, mkpts1_ = map(lambda x: x[kpts_indices],[mkpts0_,mkpts1_])
        colors=[]
        for i in range(len(mkpts0_)):
            colors.append(color)
        plot_matches(mkpts0_, mkpts1_, colors, lw=0.1, ps=0)
        
    if draw_local_window is not None:
        for type in draw_local_window:
            if '0_a' in type:
                assert keypoints_more_0_a is not None
                plot_local_windows(keypoints_more_0_a,color='r',lw=0.1,ax_=0,window_size=window_size)
            if '0_b' in type:
                assert keypoints_more_0_b is not None
                plot_local_windows(keypoints_more_0_b,color='r',lw=0.1,ax_=0,window_size=window_size)
            if '1_a' in type:
                assert keypoints_more_1_a is not None
                plot_local_windows(keypoints_more_1_a,color='r',lw=0.1,ax_=1,window_size=window_size)
                
    fig = plt.gcf()
    txt_color = 'k' if image0[:100, :150].mean() > 200 else 'w'
    fig.text(
        0.01, 0.99, '\n'.join(text), transform=fig.axes[0].transAxes,
        fontsize=15, va='top', ha='left', color=txt_color)

    txt_color = 'k' if image0[-100:, :150].mean() > 200 else 'w'
    fig.text(
        0.01, 0.01, '\n'.join(small_text), transform=fig.axes[0].transAxes,
        fontsize=5, va='bottom', ha='left', color=txt_color)
    if path:
        plt.savefig(str(path), bbox_inches='tight', pad_inches=0)
        #plt.close()
    # TODO: Would it leads to any issue without current figure opened?
    return fig


def make_matching_plot_fast(image0, image1, kpts0, kpts1, mkpts0,
                            mkpts1, color, text, path=None,
                            show_keypoints=False, margin=10,
                            opencv_display=False, opencv_title='',
                            small_text=[]):
    H0, W0 = image0.shape
    H1, W1 = image1.shape
    H, W = max(H0, H1), W0 + W1 + margin

    out = 255*np.ones((H, W), np.uint8)
    out[:H0, :W0] = image0
    out[:H1, W0+margin:] = image1
    out = np.stack([out]*3, -1)

    if show_keypoints:
        kpts0, kpts1 = np.round(kpts0).astype(int), np.round(kpts1).astype(int)
        white = (255, 255, 255)
        black = (0, 0, 0)
        for x, y in kpts0:
            cv2.circle(out, (x, y), 2, black, -1, lineType=cv2.LINE_AA)
            cv2.circle(out, (x, y), 1, white, -1, lineType=cv2.LINE_AA)
        for x, y in kpts1:
            cv2.circle(out, (x + margin + W0, y), 2, black, -1,
                       lineType=cv2.LINE_AA)
            cv2.circle(out, (x + margin + W0, y), 1, white, -1,
                       lineType=cv2.LINE_AA)

    mkpts0, mkpts1 = np.round(mkpts0).astype(int), np.round(mkpts1).astype(int)
    color = (np.array(color[:, :3])*255).astype(int)[:, ::-1]
    for (x0, y0), (x1, y1), c in zip(mkpts0, mkpts1, color):
        c = c.tolist()
        cv2.line(out, (x0, y0), (x1 + margin + W0, y1),
                 color=c, thickness=1, lineType=cv2.LINE_AA)
        # display line end-points as circles
        cv2.circle(out, (x0, y0), 2, c, -1, lineType=cv2.LINE_AA)
        cv2.circle(out, (x1 + margin + W0, y1), 2, c, -1,
                   lineType=cv2.LINE_AA)

    # Scale factor for consistent visualization across scales.
    sc = min(H / 640., 2.0)

    # Big text.
    Ht = int(30 * sc)  # text height
    txt_color_fg = (255, 255, 255)
    txt_color_bg = (0, 0, 0)
    for i, t in enumerate(text):
        cv2.putText(out, t, (int(8*sc), Ht*(i+1)), cv2.FONT_HERSHEY_DUPLEX,
                    1.0*sc, txt_color_bg, 2, cv2.LINE_AA)
        cv2.putText(out, t, (int(8*sc), Ht*(i+1)), cv2.FONT_HERSHEY_DUPLEX,
                    1.0*sc, txt_color_fg, 1, cv2.LINE_AA)

    # Small text.
    Ht = int(18 * sc)  # text height
    for i, t in enumerate(reversed(small_text)):
        cv2.putText(out, t, (int(8*sc), int(H-Ht*(i+.6))), cv2.FONT_HERSHEY_DUPLEX,
                    0.5*sc, txt_color_bg, 2, cv2.LINE_AA)
        cv2.putText(out, t, (int(8*sc), int(H-Ht*(i+.6))), cv2.FONT_HERSHEY_DUPLEX,
                    0.5*sc, txt_color_fg, 1, cv2.LINE_AA)

    if path is not None:
        cv2.imwrite(str(path), out)

    if opencv_display:
        cv2.imshow(opencv_title, out)
        cv2.waitKey(1)

    return out