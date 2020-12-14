import numpy as np

try:
    import matplotlib.pyplot as plt
except:
    plt = None
from flopy.plot import plotutil
from flopy.utils import geometry
import warnings


class ZCrossSection(object):
    """
    Development and Replacement class for current flopy cross section
    code.


    Parameters:
    ----------

    """
    def __init__(self,
                 ax=None,
                 model=None,
                 modelgrid=None,
                 line=None,
                 extent=None,
                 geographic_coords=False):

        self.ax = ax
        self.geographic_coords = geographic_coords
        if plt is None:
            s = (
                    "Could not import matplotlib.  Must install matplotlib "
                    + " in order to use ModelCrossSection method"
            )
            raise ImportError(s)

        self.model = model

        if model is not None:
            self.mg = model.modelgrid

        elif modelgrid is not None:
            self.mg = modelgrid
            if self.mg is None:
                raise AssertionError("Cannot find model grid ")

        else:
            raise Exception("Cannot find model grid")

        if self.mg.top is None or self.mg.botm is None:
            raise AssertionError("modelgrid top and botm must be defined")

        linekeys = [linekeys.lower() for linekeys in list(line.keys())]

        if len(linekeys) != 1:
            s = "only row, column, or line can be specified in line dictionary.\n"
            s += "keys specified: "
            for k in linekeys:
                s += "{} ".format(k)
            raise AssertionError(s)

        if ax is None:
            self.ax = plt.gca()
        else:
            self.ax = ax

        onkey = list(line.keys())[0]
        eps = 1.0e-4
        xedge, yedge = self.mg.xyedges
        self.__geographic_xpts = None

        # un-translate model grid into model coordinates
        self.xcellcenters, self.ycellcenters = geometry.transform(
            self.mg.xcellcenters,
            self.mg.ycellcenters,
            self.mg.xoffset,
            self.mg.yoffset,
            self.mg.angrot_radians,
            inverse=True,
        )

        try:
            self.xvertices, self.yvertices = geometry.transform(
                self.mg.xvertices,
                self.mg.yvertices,
                self.mg.xoffset,
                self.mg.yoffset,
                self.mg.angrot_radians,
                inverse=True)
        except ValueError:
            # irregular shapes in vertex grid ie. squares and triangles
            xverts, yverts = \
                plotutil.UnstructuredPlotUtilities.irregular_shape_patch(
                self.mg.xvertices, self.mg.yvertices)

            self.xvertices, self.yvertices = geometry.transform(
                xverts,
                yverts,
                self.mg.xoffset,
                self.mg.yoffset,
                self.mg.angrot_radians,
                inverse=True)

        if onkey == 'row':
            ycenter = self.ycellcenters.T[0]
            pts = [
                (xedge[0] + eps, ycenter[int(line[onkey])] - eps),
                (xedge[-1] - eps, ycenter[int(line[onkey])] + eps),
            ]
        elif onkey == 'column':
            xcenter = self.xcellcenters[0, :]
            pts = [
                (xcenter[int(line[onkey])] + eps, yedge[0] - eps),
                (xcenter[int(line[onkey])] - eps, yedge[-1] + eps),
            ]
        else:
            verts = line[onkey]
            xp = []
            yp = []
            for [v1, v2] in verts:
                xp.append(v1)
                yp.append(v2)

            xp, yp = self.mg.get_local_coords(xp, yp)
            pts = [(xt, yt) for xt, yt in zip(xp, yp)]

        self.pts = np.array(pts)

        self.xypts = plotutil.UnstructuredPlotUtilities.line_intersect_grid(
            self.pts, self.xvertices, self.yvertices
        )

        if len(self.xypts) < 2:
            s = "cross-section cannot be created\n."
            s += "   less than 2 points intersect the model grid\n"
            s += "   {} points intersect the grid.".format(len(self.xypts))
            raise Exception(s)

        if self.geographic_coords:
            # transform back to geographic coordinates
            xypts = {}
            for nn, pt in self.xypts.items():
                xp = [t[0] for t in pt]
                yp = [t[1] for t in pt]
                xp, yp = geometry.transform(
                    xp,
                    yp,
                    self.mg.xoffset,
                    self.mg.yoffset,
                    self.mg.angrot_radians,
                )
                xypts[nn] = [(xt, yt) for xt, yt in zip(xp, yp)]

            self.xypts = xypts

        top = self.mg.top
        top.shape = (1,) + top.shape[1:]
        botm = self.mg.botm

        self.elev = np.concatenate((top, botm), axis=0)

        self.idomain = self.mg.idomain
        if self.mg.idomain is None:
            self.idomain = np.ones(botm.shape, dtype=int)

        # choose a projection direction based on maximum information
        xpts = []
        ypts = []
        for nn, verts in self.xypts.items():
            for v in verts:
                xpts.append(v[0])
                ypts.append(v[1])

        if np.max(xpts) - np.min(xpts) > np.max(ypts) - np.min(ypts):
            self.direction = "x"
        else:
            self.direction = "y"

        # make vertex array based on projection direction
        self.projpts = self.set_zpts(None)

        # Create cross-section extent
        if extent is None:
            self.extent = self.get_extent()
        else:
            self.extent = extent

        self.layer0 = None
        self.layer1 = None

        self.d = {
            i: (np.min(np.array(v).T[0]), np.max(np.array(v).T[0]))
            for i, v in sorted(self.projpts.items())
        }

        self.xpts = None
        self.active = None
        self.ncb = None
        self.laycbd = None
        self.zpts = None
        self.xcentergrid = None
        self.zcentergrid = None
        self.geographic_xcentergrid = None
        self.geographic_xpts = None

        # Set axis limits
        self.ax.set_xlim(self.extent[0], self.extent[1])
        self.ax.set_ylim(self.extent[2], self.extent[3])


    def get_extent(self):
        """
        Get the extent of the rotated and offset grid

        Returns
        -------
        tuple : (xmin, xmax, ymin, ymax)
        """
        xpts = []
        for _, verts in self.projpts.items():
            for v in verts:
                xpts.append(v[0])

        xmin = np.min(xpts)
        xmax = np.max(xpts)

        ymin = np.min(self.elev)
        ymax = np.max(self.elev)

        return (xmin, xmax, ymin, ymax)