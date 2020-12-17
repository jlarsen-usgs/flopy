import numpy as np

try:
    import matplotlib.pyplot as plt
    import matplotlib.colors
    from matplotlib.patches import Polygon
except:
    plt = None

from flopy.plot import plotutil
from flopy.utils import geometry
import copy
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

        if not isinstance(line, dict):
            raise AssertionError("A line dictionary must be provided")

        line = {k.lower(): v for k, v in line.items()}

        if len(line) != 1:
            s = "only row, column, or line can be specified in line dictionary.\n"
            s += "keys specified: "
            for k in line.keys():
                s += "{} ".format(k)
            raise AssertionError(s)

        if ax is None:
            self.ax = plt.gca()
        else:
            self.ax = ax

        onkey = list(line.keys())[0]
        self.__geographic_xpts = None

        # un-translate model grid into model coordinates
        xcellcenters, ycellcenters = geometry.transform(
            self.mg.xcellcenters,
            self.mg.ycellcenters,
            self.mg.xoffset,
            self.mg.yoffset,
            self.mg.angrot_radians,
            inverse=True,
        )

        xverts, yverts = self.mg.cross_section_vertices

        xverts, yverts = \
            plotutil.UnstructuredPlotUtilities.irregular_shape_patch(
            xverts, yverts)

        self.xvertices, self.yvertices = geometry.transform(
            xverts,
            yverts,
            self.mg.xoffset,
            self.mg.yoffset,
            self.mg.angrot_radians,
            inverse=True)

        if onkey in ('row', 'column'):
            eps = 1.0e-4
            xedge, yedge = self.mg.xyedges
            if onkey == 'row':
                ycenter = ycellcenters.T[0]
                pts = [
                    (xedge[0] - eps, ycenter[int(line[onkey])]),
                    (xedge[-1] + eps, ycenter[int(line[onkey])]),
                ]
            else:
                xcenter = xcellcenters[0, :]
                pts = [
                    (xcenter[int(line[onkey])], yedge[0] + eps),
                    (xcenter[int(line[onkey])], yedge[-1] - eps),
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


        top = self.mg.top.reshape(1, self.mg.ncpl)
        botm = self.mg.botm.reshape(self.mg.nlay, self.mg.ncpl)

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

        # todo: might be able to remove this at some point...
        self.d = {
            i: (np.min(np.array(v).T[0]), np.max(np.array(v).T[0]))
            for i, v in sorted(self.projpts.items())
        }

        # this is actually x or y based on projection
        self.xcenters = [
            np.mean(np.array(v).T[0]) for i, v in sorted(self.projpts.items())
        ]

        self._polygons = {}

        # Set axis limits
        self.ax.set_xlim(self.extent[0], self.extent[1])
        self.ax.set_ylim(self.extent[2], self.extent[3])

    @property
    def polygons(self):
        """
        Method to return cached matplotlib polygons for a cross
        section

        Returns
        -------
            dict : [matplotlib.patches.Polygon]
        """
        if not self._polygons:
            for cell, verts in self.projpts.items():
                verts = \
                    plotutil.UnstructuredPlotUtilities.arctan2(np.array(verts))

                self._polygons[cell] = Polygon(verts, closed=True)

        return copy.copy(self._polygons)

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

    def plot_array(self, a, masked_values=None, head=None, **kwargs):
        """
        Plot a three-dimensional array as a patch collection.

        Parameters
        ----------
        a : numpy.ndarray
            Three-dimensional array to plot.
        masked_values : iterable of floats, ints
            Values to mask.
        head : numpy.ndarray
            Three-dimensional array to set top of patches to the minimum
            of the top of a layer or the head value. Used to create
            patches that conform to water-level elevations.
        **kwargs : dictionary
            keyword arguments passed to matplotlib.collections.PatchCollection

        Returns
        -------
        patches : matplotlib.collections.PatchCollection

        """
        if "ax" in kwargs:
            ax = kwargs.pop("ax")
        else:
            ax = self.ax

        if not isinstance(a, np.ndarray):
            a = np.array(a)

        if a.ndim > 1:
            a = np.ravel(a)

        if masked_values is not None:
            for mval in masked_values:
                a = np.ma.masked_values(a, mval)

        if isinstance(head, np.ndarray):
            projpts = self.set_zpts(np.ravel(head))
        else:
            projpts = None

        pc = self.get_grid_patch_collection(a, projpts, **kwargs)
        if pc is not None:
            ax.add_collection(pc)
            ax.set_xlim(self.extent[0], self.extent[1])
            ax.set_ylim(self.extent[2], self.extent[3])

        return pc

    def contour_array(self, a, masked_values=None, head=None, **kwargs):
        """
        Contour a two-dimensional array.

        Parameters
        ----------
        a : numpy.ndarray
            Three-dimensional array to plot.
        masked_values : iterable of floats, ints
            Values to mask.
        head : numpy.ndarray
            Three-dimensional array to set top of patches to the minimum
            of the top of a layer or the head value. Used to create
            patches that conform to water-level elevations.
        **kwargs : dictionary
            keyword arguments passed to matplotlib.pyplot.contour

        Returns
        -------
        contour_set : matplotlib.pyplot.contour

        """
        if plt is None:
            err_msg = (
                "matplotlib must be installed to " + "use contour_array()"
            )
            raise ImportError(err_msg)
        else:
            import matplotlib.tri as tri

        if not isinstance(a, np.ndarray):
            a = np.array(a)

        if a.ndim > 1:
            a = np.ravel(a)

        if "ax" in kwargs:
            ax = kwargs.pop("ax")
        else:
            ax = self.ax

        xcenters = self.xcenters
        plotarray = np.array([a[cell] for cell in sorted(self.projpts)])

        if self.mg.nlay == 1 and not isinstance(self.mg.ncpl, np.ndarray):
            zcenters = []
            if isinstance(head, np.ndarray):
                head = head.reshape(1, self.mg.ncpl)
                head = np.vstack((head, head))
            else:
                head = self.elev.reshape(2, self.mg.ncpl)

            elev = self.elev.reshape(2, self.mg.ncpl)
            for k, ev in enumerate(elev):
                if k == 0:
                    zc = [ev[i] if head[k][i] > ev[i] else head[k][i]
                          for i in sorted(self.projpts)]
                else:
                    zc = [ev[i] for i in sorted(self.projpts)]
                zcenters.append(zc)

            plotarray = np.vstack((plotarray, plotarray))
            xcenters = np.vstack((xcenters, xcenters))
            zcenters = np.array(zcenters)

        else:
            if isinstance(head, np.ndarray):
                zcenters = self.set_zcentergrid(np.ravel(head))
            else:
                zcenters = np.array([
                    np.mean(np.array(v).T[1])
                    for i, v in sorted(self.projpts.items())
                    ])

        # work around for tri-contour ignore vmin & vmax
        # necessary for the tri-contour NaN issue fix
        if "levels" not in kwargs:
            if "vmin" not in kwargs:
                vmin = np.nanmin(plotarray)
            else:
                vmin = kwargs.pop("vmin")
            if "vmax" not in kwargs:
                vmax = np.nanmax(plotarray)
            else:
                vmax = kwargs.pop("vmax")

            levels = np.linspace(vmin, vmax, 7)
            kwargs["levels"] = levels

        # workaround for tri-contour nan issue
        plotarray[np.isnan(plotarray)] = -(2 ** 31)
        if masked_values is None:
            masked_values = [-(2 ** 31)]
        else:
            masked_values = list(masked_values)
            if -(2 ** 31) not in masked_values:
                masked_values.append(-(2 ** 31))

        ismasked = None
        if masked_values is not None:
            for mval in masked_values:
                if ismasked is None:
                    ismasked = np.isclose(plotarray, mval)
                else:
                    t = np.isclose(plotarray, mval)
                    ismasked += t

        plot_triplot = False
        if "plot_triplot" in kwargs:
            plot_triplot = kwargs.pop("plot_triplot")

        if "extent" in kwargs:
            extent = kwargs.pop("extent")

            idx = (
                (xcenters >= extent[0])
                & (xcenters <= extent[1])
                & (zcenters >= extent[2])
                & (zcenters <= extent[3])
            )
            plotarray = plotarray[idx].flatten()
            xcenters = xcenters[idx].flatten()
            zcenters = zcenters[idx].flatten()

        if self.mg.nlay == 1 and not isinstance(self.mg.ncpl, np.ndarray):
            plotarray = np.ma.masked_array(plotarray, ismasked)
            contour_set = ax.contour(xcenters, zcenters, plotarray, **kwargs)
        else:
            triang = tri.Triangulation(xcenters, zcenters)

            if ismasked is not None:
                ismasked = ismasked.flatten()
                mask = np.any(
                    np.where(ismasked[triang.triangles], True, False), axis=1
                )
                triang.set_mask(mask)

            contour_set = ax.tricontour(triang, plotarray, **kwargs)
            if plot_triplot:
                ax.triplot(triang, color="black", marker="o", lw=0.75)

        ax.set_xlim(self.extent[0], self.extent[1])
        ax.set_ylim(self.extent[2], self.extent[3])

        return contour_set


    def plot_inactive(self, ibound=None, color_noflow="black", **kwargs):
        """
        Make a plot of inactive cells.  If not specified, then pull ibound
        from the self.ml

        Parameters
        ----------
        ibound : numpy.ndarray
            ibound array to plot.  (Default is ibound in 'BAS6' package.)

        color_noflow : string
            (Default is 'black')

        Returns
        -------
        quadmesh : matplotlib.collections.QuadMesh

        """
        if ibound is None:
            if self.mg.idomain is None:
                raise AssertionError("An idomain array must be provided")
            else:
                ibound = self.mg.idomain

        plotarray = np.zeros(ibound.shape, dtype=np.int)
        idx1 = ibound == 0
        plotarray[idx1] = 1
        plotarray = np.ma.masked_equal(plotarray, 0)
        cmap = matplotlib.colors.ListedColormap(["0", color_noflow])
        bounds = [0, 1, 2]
        norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)
        patches = self.plot_array(plotarray, cmap=cmap, norm=norm, **kwargs)

        return patches

    def plot_ibound(
        self,
        ibound=None,
        color_noflow="black",
        color_ch="blue",
        color_vpt="red",
        head=None,
        **kwargs
    ):
        """
        Make a plot of ibound.  If not specified, then pull ibound from the
        self.model

        Parameters
        ----------
        ibound : numpy.ndarray
            ibound array to plot.  (Default is ibound in 'BAS6' package.)
        color_noflow : string
            (Default is 'black')
        color_ch : string
            Color for constant heads (Default is 'blue'.)
        head : numpy.ndarray
            Three-dimensional array to set top of patches to the minimum
            of the top of a layer or the head value. Used to create
            patches that conform to water-level elevations.
        **kwargs : dictionary
            keyword arguments passed to matplotlib.collections.PatchCollection

        Returns
        -------
        patches : matplotlib.collections.PatchCollection

        """
        if ibound is None:
            if self.model is not None:
                if self.model.version == "mf6":
                    color_ch = color_vpt

            if self.mg.idomain is None:
                raise AssertionError("Ibound/Idomain array must be provided")

            ibound = self.mg.idomain

        plotarray = np.zeros(ibound.shape, dtype=np.int)
        idx1 = ibound == 0
        idx2 = ibound < 0
        plotarray[idx1] = 1
        plotarray[idx2] = 2
        plotarray = np.ma.masked_equal(plotarray, 0)
        cmap = matplotlib.colors.ListedColormap(
            ["none", color_noflow, color_ch]
        )
        bounds = [0, 1, 2, 3]
        norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)
        # mask active cells
        patches = self.plot_array(
            plotarray,
            masked_values=[0],
            head=head,
            cmap=cmap,
            norm=norm,
            **kwargs
        )
        return patches

    def plot_grid(self, **kwargs):
        """
        Plot the grid lines.

        Parameters
        ----------
            kwargs : ax, colors.  The remaining kwargs are passed into the
                the LineCollection constructor.

        Returns
        -------
            lc : matplotlib.collections.LineCollection

        """
        if "ax" in kwargs:
            ax = kwargs.pop("ax")
        else:
            ax = self.ax

        col = self.get_grid_line_collection(**kwargs)
        if col is not None:
            ax.add_collection(col)
            ax.set_xlim(self.extent[0], self.extent[1])
            ax.set_ylim(self.extent[2], self.extent[3])

        return col

    def get_grid_line_collection(self, **kwargs):
        """
        Get a LineCollection of the grid

        Parameters
        ----------
        **kwargs : dictionary
            keyword arguments passed to matplotlib.collections.LineCollection

        Returns
        -------
        linecollection : matplotlib.collections.LineCollection
        """
        if plt is None:
            err_msg = (
                "matplotlib must be installed to "
                + "use get_grid_line_collection()"
            )
            raise ImportError(err_msg)
        else:
            from matplotlib.patches import Polygon
            from matplotlib.collections import PatchCollection

        color = "grey"
        if "ec" in kwargs:
            color = kwargs.pop("ec")
        if "color" in kwargs:
            color = kwargs.pop("color")

        polygons = [p for _, p in sorted(self.polygons.items())]
        if len(polygons) > 0:
            patches = PatchCollection(
                polygons, edgecolor=color, facecolor="none", **kwargs
            )
        else:
            patches = None

        return patches

    def set_zpts(self, vs):
        """
        Get an array of projected vertices corrected with corrected
        elevations based on minimum of cell elevation (self.elev) or
        passed vs numpy.ndarray

        Parameters
        ----------
        vs : numpy.ndarray
            Two-dimensional array to plot.

        Returns
        -------
        zpts : dict

        """
        # make vertex array based on projection direction
        if vs is not None:
            if not isinstance(vs, np.ndarray):
                vs = np.array(vs)

        if self.direction == "x":
            xyix = 0
        else:
            xyix = -1

        projpts = {}

        nlay = 1
        if len(self.xvertices) != self.mg.nnodes:
            nlay = self.mg.nlay

        for k in range(1, nlay + 1):
            top = self.elev[k - 1, :]
            botm = self.elev[k, :]
            adjnn = (k - 1) * self.mg.ncpl
            d0 = 0
            for nn, verts in sorted(
                self.xypts.items(), key=lambda q: q[-1][xyix][xyix]
            ):
                if vs is None:
                    t = top[nn]
                else:
                    t = vs[nn]
                    if top[nn] < vs[nn]:
                        t = top[nn]

                b = botm[nn]
                if self.geographic_coords:
                    if self.direction == "x":
                        projt = [(v[0], t) for v in verts]
                        projb = [(v[0], b) for v in verts]
                    else:
                        projt = [(v[1], t) for v in verts]
                        projb = [(v[1], b) for v in verts]
                else:
                    verts = np.array(verts).T
                    a2 = (np.max(verts[0]) - np.min(verts[0])) ** 2
                    b2 = (np.max(verts[1]) - np.min(verts[1])) ** 2
                    c = np.sqrt(a2 + b2)
                    d1 = d0 + c
                    projt = [(d0, t), (d1, t)]
                    projb = [(d0, b), (d1, b)]
                    d0 += c

                projpts[nn + adjnn] = projt + projb

        return projpts

    def set_zcentergrid(self, vs, kstep=1):
        """
        Get an array of z elevations at the center of a cell that is based
        on minimum of cell top elevation (self.elev) or passed vs numpy.ndarray

        Parameters
        ----------
        vs : numpy.ndarray
            Three-dimensional array to plot.

        Returns
        -------
        zcentergrid : numpy.ndarray

        """
        verts = self.set_zpts(vs)
        zcenters = [
            np.mean(np.array(v).T[1])
            for i, v in sorted(verts.items())
            if (i // self.mg.ncpl) % kstep == 0
        ]
        return zcenters

    def get_grid_patch_collection(self, plotarray, projpts=None, **kwargs):
        """
        Get a PatchCollection of plotarray in unmasked cells

        Parameters
        ----------
        plotarray : numpy.ndarray
            One-dimensional array to attach to the Patch Collection.
         projpts : dict
            dictionary defined by node number which contains model
            patch vertices.
        **kwargs : dictionary
            keyword arguments passed to matplotlib.collections.PatchCollection

        Returns
        -------
        patches : matplotlib.collections.PatchCollection

        """
        use_cache = False
        if projpts is None:
            use_cache = True
            projpts = self.polygons

        if plt is None:
            err_msg = (
                    "matplotlib must be installed to "
                    + "use get_grid_patch_collection()"
            )
            raise ImportError(err_msg)
        else:
            from matplotlib.patches import Polygon
            from matplotlib.collections import PatchCollection

        if "vmin" in kwargs:
            vmin = kwargs.pop("vmin")
        else:
            vmin = None
        if "vmax" in kwargs:
            vmax = kwargs.pop("vmax")
        else:
            vmax = None

        rectcol = []
        data = []
        for cell, poly in sorted(projpts.items()):
            if not use_cache:
                poly = \
                    plotutil.UnstructuredPlotUtilities.arctan2(np.array(poly))

            if np.isnan(plotarray[cell]):
                continue
            elif plotarray[cell] is np.ma.masked:
                continue

            if use_cache:
                rectcol.append(poly)
            else:
                rectcol.append(Polygon(poly, closed=True))
            data.append(plotarray[cell])

        if len(rectcol) > 0:
            patches = PatchCollection(rectcol, **kwargs)
            patches.set_array(np.array(data))
            patches.set_clim(vmin, vmax)

        else:
            patches = None

        return patches