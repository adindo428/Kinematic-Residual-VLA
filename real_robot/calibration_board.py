"""Plain chessboard geometry and stable image-space labels. No robot I/O."""
import cv2
import numpy as np


def contact_layout(squares=(6,6)):
    nx,ny=map(int,squares)
    if min(nx,ny)<5 or max(nx,ny)>20:
        raise ValueError('Require 5..20 squares per side, counting squares rather than inner corners')
    cols,rows=nx-1,ny-1
    fit=(0,cols-1,(rows-1)*cols,rows*cols-1)
    ix,iy=max(1,(cols-1)//4),max(1,(rows-1)//4)
    check=(iy*cols+ix,(rows-1-iy)*cols+(cols-1-ix))
    targets=[(placement,corner,split) for placement in range(2)
             for split,ids in [('fit',fit),('check',check)] for corner in ids]
    return dict(fit_ids=fit,check_ids=check,targets=targets,
                low_pitch_ids=(fit[0],fit[2],check[0]),corner_count=rows*cols)


class Chessboard:
    def __init__(self,squares,square_m):
        self.squares=tuple(map(int,squares))
        contact_layout(self.squares)
        if not np.isfinite(square_m) or square_m<=0:raise ValueError('Square size must be positive')
        self.pattern_size=(self.squares[0]-1,self.squares[1]-1)
        cols,rows=self.pattern_size
        self.points=np.array([[(col+1)*square_m,(row+1)*square_m,0.]
                              for row in range(rows) for col in range(cols)],np.float32)

    def getChessboardCorners(self):
        return self.points.copy()


class ChessboardDetector:
    """Return the same corner interface as CharucoDetector, with viewer-only IDs.

    A complete pattern is required. Match equivalent flips/transposes against
    the initial/locked grid so detector ordering flips cannot relabel targets.
    This does not make a symmetric checkerboard a uniquely marked fiducial.
    """
    def __init__(self,board):
        self.board=board
        self.anchor_grid=None
        self.last_scale=1

    def clear_reference(self):
        self.anchor_grid=None

    def lock_reference(self,pixels):
        cols,rows=self.board.pattern_size
        self.anchor_grid=np.asarray(pixels,np.float32).reshape(rows,cols,2).copy()

    def order_corners(self,corners):
        cols,rows=self.board.pattern_size
        grid=np.asarray(corners,np.float32).reshape(rows,cols,2)
        variants=[grid,grid[::-1],grid[:,::-1],grid[::-1,::-1]]
        if rows==cols:
            transposed=grid.transpose(1,0,2)
            variants += [transposed,transposed[::-1],transposed[:,::-1],transposed[::-1,::-1]]
        if self.anchor_grid is None:
            # Pick an image-near upper-left origin, with the first row running
            # as far right as possible. Later frames match this initial grid.
            selected=min(variants,key=lambda g:(float(g[0,0].sum()),-float(g[0,-1,0]-g[0,0,0])))
            self.anchor_grid=selected.copy()
        else:
            selected=min(variants,key=lambda g:float(np.mean(np.sum((g-self.anchor_grid)**2,axis=2))))
        return np.ascontiguousarray(selected.reshape(-1,1,2))

    def detectBoard(self,image):
        gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY) if image.ndim==3 else image
        flags=cv2.CALIB_CB_ADAPTIVE_THRESH|cv2.CALIB_CB_NORMALIZE_IMAGE|cv2.CALIB_CB_FAST_CHECK
        found,corners=cv2.findChessboardCorners(gray,self.board.pattern_size,flags)
        if found:
            corners=cv2.cornerSubPix(gray,corners,(3,3),(-1,-1),
                         (cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_MAX_ITER,30,.001))
        else:
            found,corners=cv2.findChessboardCornersSB(gray,self.board.pattern_size,cv2.CALIB_CB_NORMALIZE_IMAGE)
        if not found:return None,None,[],None
        corners=self.order_corners(corners)
        ids=np.arange(len(corners),dtype=np.int32).reshape(-1,1)
        return corners,ids,[],None
