import os, pdb
import numpy as np
import xarray as xr
import gsw
from tqdm import tqdm
from configparser import ConfigParser
from matplotlib import pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cf

class Model:
    def __init__(self, zplot_1: float, zplot_2: float, config_file: str = 'config.ini') -> None:
        self.config = ConfigParser()
        self.config.read(config_file)
        self.zplot_1 = zplot_1
        self.zplot_2 = zplot_2
        self.__getExecutionParams()
        self.__checkInfra()
        self.__getEnvironmentalParams()
        self.__wind()
        
        self.__loadGrid()

    def __checkInfra(self) -> None:
        '''
        Verifies if the directory structure is correct.
        If it isn't, it will create the missing directories.
        '''
        self.base_dir = 'experiments/'+self.experiment_name
        if not os.path.exists('experiments'):
            os.makedirs('experiments')
        if not os.path.exists(self.base_dir):
            os.makedirs(self.base_dir)
        if not os.path.exists(self.base_dir + '/out'):
            os.makedirs(self.base_dir + '/out')
        if not os.path.exists(self.base_dir + '/fig'):
            os.makedirs(self.base_dir + '/fig')

        self.img_dir = self.base_dir + '/fig'
        self.field_dir = self.base_dir + '/out'

    def __getEnvironmentalParams(self) -> None:
        '''
        Reads the environmental parameters from the config file.
        '''
        params = self.config['ENVIRONMENTAL']
        self.g = float(params['g'])
        self.difh = float(params['difh'])
        self.difz = float(params['difz'])
        self.r = float(params['r'])
        self.rho = float(params['rho'])
        self.rho_air = float(params['rho_air'])
        self.fric = float(params['fric'])

    def __getExecutionParams(self) -> None:
        '''
        Reads the execution parameters from the config file.
        '''
        params = self.config['EXECUTION']
        self.experiment_name = params['experiment']
        self.dz = float(params['dz'])
        self.dt = float(params['dt'])
        self.nmax = int(params['nmax'])
        self.freqplot = int(params['freqplot'])

    def __loadGrid(self) -> None:
        '''
        Loads the grid file
        '''
        path = "grid/"+self.config['EXECUTION']['grid']
        grid = xr.open_dataset(path)
        self.dx = grid.attrs['dx']
        self.dy = grid.attrs['dy']
        self.mean_lat = grid.attrs['mean_lat']
        self.f0 = gsw.f(self.mean_lat)
        self.lon = grid.lon.values
        self.lat = grid.lat.values

        self.x = grid.x.values
        self.y = grid.y.values
        self.bat = grid.elev.values

        self.batu = self.bat[:-1:2,1::2]
        self.batv = self.bat[1::2,:-1:2]
        self.bate = self.bat[1::2, 1::2]

        self.kmaru = self.batu.copy()*0
        self.kmaru[self.batu>0] = 1
        self.kmarv = self.batv.copy()*0
        self.kmarv[self.batv>0] = 1
        self.kmare = self.bate.copy()*0
        self.kmare[self.bate>0] = 1

    def __wind(self) -> None:
        params = self.config['ENVIRONMENTAL']
        uwind = float(params['uwind'])
        vwind = float(params['vwind'])
        wwind = np.sqrt(uwind**2 + vwind**2)
        self.taux = self.fric*self.rho_air*uwind*wwind
        self.tauy = self.fric*self.rho_air*vwind*wwind

    def __continuity(self, k: int, j: int, eta0: np.ndarray, u1: np.ndarray, v1: np.ndarray) -> np.ndarray:
        forcx = 0
        forcy = 0 
        for l in range(1, self.lmax-1):
            forcx += (u1[k,j+1,l] - u1[k,j,l])*self.dz/self.dx
            forcy += (v1[k,j,l] - v1[k-1,j,l])*self.dz/self.dy
        return eta0[k,j] - self.dt*(forcx + forcy)
    
    def __moment_x(self, k: int, j: int, l: int, u0: np.ndarray, u1: np.ndarray,
                   v1: np.ndarray, eta1: np.ndarray) -> np.ndarray:
        dudzinf = (u0[k,j,l+1] - u0[k,j,l])*self.difzdz
        dudzsup = (u0[k,j,l] - u0[k,j,l-1])*self.difzdz
        if l == 1:
            dudzsup = -self.taux/self.rho
        umed = (u1[k,j-1,l]+2*u1[k,j,l]+u1[k,j+1,l])/4
        vmedu = (v1[k,j,l]+v1[k,j-1,l]+v1[k-1,j-1,l]+v1[k-1,j,l])/4
        
        coriolis = self.f0*vmedu
        gpres = -self.g*(eta1[k,j] - eta1[k,j-1])/self.dx
        difvert = (dudzinf - dudzsup)/self.dz
        decaimento = -self.r*u0[k,j,l]
        advx = -umed*(u1[k,j+1,l] - u1[k,j-1,l])/self.dx2
        advy = -vmedu*(u1[k+1,j,l] - u1[k-1,j,l])/self.dy2
        tdifx = self.dif2x*(u0[k,j+1,l] - 2*u0[k,j,l] + u0[k,j-1,l])
        tdify = self.dif2y*(u0[k+1,j,l] - 2*u0[k,j,l] + u0[k-1,j,l])
        
        forc = coriolis + gpres + difvert + decaimento + advx +\
                advy + tdifx + tdify
        return u0[k,j,l] + forc*self.dt

    def __moment_y(self, k: int, j: int, l: int, v0: np.ndarray, v1: np.ndarray,
                   u1: np.ndarray, eta1: np.ndarray) -> np.ndarray:
        dvdzinf = (v0[k,j,l+1] - v0[k,j,l])*self.difzdz
        dvdzsup = (v0[k,j,l] - v0[k,j,l-1])*self.difzdz
        if l == 1:
            dvdzsup = -self.tauy/self.rho
        vmed = (v1[k-1,j,l]+v1[k,j,l]*2+v1[k+1,j,l])/4
        umedv = (u1[k,j,l]+u1[k+1,j,l]+u1[k+1,j+1,l]+u1[k,j+1,l])/4
        
        coriolis = -self.f0*umedv
        gpres = -self.g*(eta1[k+1,j] - eta1[k,j])/self.dy
        difvert = (dvdzinf - dvdzsup)/self.dz
        decaimento = -self.r*v0[k,j,l]
        advy = -vmed*(v1[k+1,j,l] - v1[k-1,j,l])/self.dy2
        advx = -umedv*(v1[k,j+1,l] - v1[k,j-1,l])/self.dx2
        tdifx = self.dif2x*(v0[k,j+1,l] - 2*v0[k,j,l] + v0[k,j-1,l])
        tdify = self.dif2y*(v0[k+1,j,l] - 2*v0[k,j,l] + v0[k-1,j,l])
        
        forc = coriolis + gpres + difvert + decaimento + advy +\
                advx + tdifx + tdify
        
        return v0[k,j,l] + forc*self.dt

    def run(self) -> None:
        self.dt2 = self.dt*2
        self.dx2 = self.dx*2
        self.dy2 = self.dy*2
        self.kmax = int(len(self.y)//2)
        self.jmax = int(len(self.x)//2)
        self.lmax = int(np.ceil(np.max(self.bat))//self.dz)
        self.z = np.arange(0, self.lmax*self.dz, self.dz)

        self.difzdz = self.difz/self.dz
        self.dif2x = self.difh/self.dx/self.dx
        self.dif2y = self.difh/self.dy/self.dy

        # Grid Initialization
        eta0 = np.zeros((self.kmax, self.jmax))
        eta1 = np.zeros((self.kmax, self.jmax))
        eta2 = np.zeros((self.kmax, self.jmax))
        
        u0 = np.zeros((self.kmax, self.jmax, self.lmax))
        u1 = np.zeros((self.kmax, self.jmax, self.lmax))
        u2 = np.zeros((self.kmax, self.jmax, self.lmax))
        v0 = np.zeros((self.kmax, self.jmax, self.lmax))
        v1 = np.zeros((self.kmax, self.jmax, self.lmax))
        v2 = np.zeros((self.kmax, self.jmax, self.lmax))

        # Time Loop
        kplot = self.freqplot-1
        for n in tqdm(range(3,self.nmax+1)):
            kplot += 1
            # Continuity
            for j in range(1, self.jmax-1):
                for k in range(1, self.kmax-1):
                    if self.kmare[k,j] > 0:
                        eta2[k,j] = self.__continuity(k, j, eta0, u1, v1)
            
            # Filtering
            eta1 = ((eta0 + 2*eta1 + eta2)/4)*self.kmare
            etafil = eta1.copy()
            etafil = self.kmare*(eta0+ 2*eta1 + eta2)/4
            eta1 = etafil.copy()

            # Momentum in x-direction
            for k in range(1, self.kmax-1):
                for j in range(1, self.jmax-1):
                    if self.kmaru[k,j]*self.kmare[k,j]*self.kmare[k,j-1] > 0:
                        bat3 = np.array([self.batu[k,j], self.bate[k,j], self.bate[k,j-1]])
                        batmin = np.max(bat3)
                        lend = int(batmin//self.dz)-1
                        if lend >= u0.shape[2]: lend = int(u0.shape[2]-1)
                        for l in range(1, lend):
                            u2[k,j,l] = self.__moment_x(k, j, l, u0, u1, v1, eta1)
            
            # Filtering
            ufil = u1.copy()
            for j in range(1,self.jmax-1):
                for k in range(1, self.kmax-1):
                    ufil[k,j,:] = self.kmaru[k,j]*(u0[k,j,:] + 2*u1[k,j,:] + u2[k,j,:])/4
            u1 = ufil.copy()

            # Momentum in y-direction
            for k in range(1, self.kmax-1):
                for j in range(1, self.jmax-1):
                    if self.kmarv[k,j]*self.kmare[k,j]*self.kmare[k+1,j] > 0:
                        bat3 = np.array([self.batv[k,j], self.bate[k,j], self.bate[k+1,j]])
                        batmin = np.max(bat3)
                        lend = int(batmin//self.dz)-1
                        if lend >= v0.shape[2]: lend = int(v0.shape[2]-1)
                        for l in range(1, lend):
                            v2[k,j,l] = self.__moment_y(k, j, l, v0, v1, u1, eta1)
            
            # Filtering
            vfil = v1.copy()
            for j in range(1,self.jmax-1):
                for k in range(1, self.kmax-1):
                    vfil[k,j,:] = self.kmarv[k,j]*(v0[k,j,:] + 2*v1[k,j,:] + v2[k,j,:])/4
            v1 = vfil.copy()

            # Boundary Condition
            for j in range(self.jmax):
                eta2[0,j] = eta2[1,j]*self.kmare[0,j]
                eta2[-1,j] = eta2[-2,j]*self.kmare[-1,j]
                u2[0,j,1:-1] = u2[1,j,1:-1]*self.kmaru[0,j]
                u2[-1,j,1:-1] = u2[-2,j,1:-1]*self.kmaru[-1,j]
                v2[0,j,1:-1] = v2[1,j,1:-1]*self.kmarv[0,j]
                v2[-1,j,1:-1] = u2[-2,j,1:-1]*self.kmarv[-1,j]
            for k in range(self.kmax):
                eta2[k,0] = eta2[k,1]*self.kmare[k,0]
                eta2[k,-1] = eta2[k,-2]*self.kmare[k,-1]
                u2[k,0,1:-1] = u2[k,1,1:-1]*self.kmaru[k,0]
                u2[k,-1,1:-1] = u2[k,-2,1:-1]*self.kmaru[k,-1]
                v2[k,0,1:-1] = v2[k,1,1:-1]*self.kmarv[k,0]
                v2[k,-1,1:-1] = v2[k,-2,1:-1]*self.kmarv[k,-1]
            
            # Update
            eta0 = eta1.copy()
            eta1 = eta2.copy()
            u0 = u1.copy()
            u1 = u2.copy()
            v0 = v1.copy()
            v1 = v2.copy()

            if kplot == self.freqplot:
                kplot = 0
                z_top = np.where(self.z == self.zplot_1)[0][0]
                z_bot = np.where(self.z == self.zplot_2)[0][0]
                self.plot(f"Elevation: t = {int((n-3)*self.dt):04} s", eta2, f"/elev_{int((n-3)*self.dt):04}.png", self.z[1])
                self.plot(f"Meridional Velocity: t = {int((n-3)*self.dt):04} s\nz = {self.z[1]} m", v2[:,:,1], f"/smv_{int((n-3)*self.dt):04}.png", self.z[z_top])
                self.plot(f"Meridional Velocity: t = {int((n-3)*self.dt):04} s\nz = {self.z[z_bot]} m", v2[:,:,z_bot], f"/bmv_{int((n-3)*self.dt):04}.png", self.z[z_bot])
                self.save_field(f"field_{int((n-3)*self.dt):04}.nc", n-3, u2, v2, eta2)

    def plot_cartopy(self, title, variable, filename) -> None:
        plt.figure()
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax.set_title(title)
        cs = ax.contourf(self.lon[::2], self.lat[::2], variable)
        plt.colorbar(cs)
        ax.add_feature(cf.COASTLINE, zorder = 100)
        ax.add_feature(cf.LAND, zorder = 100)
        ax.add_feature(cf.BORDERS, zorder = 100)
        plt.tight_layout()
        plt.savefig(self.img_dir+filename)
        plt.close()

    def plot(self, title, variable, filename, z_contour: float = 0) -> None:
        plt.figure()
        plt.title(title)
        cs = plt.contourf(self.lon[::2], self.lat[::2], variable)
        plt.colorbar(cs)
        plt.contour(self.lon[::2], self.lat[::2], self.bat[::2,::2], levels=[z_contour], colors='k', linewidths=1)
        plt.tight_layout()
        plt.savefig(self.img_dir+filename)
        plt.close()

    def save_field(self, filename, n, u, v, eta) -> None:
        ds = xr.Dataset(
            coords=dict(
                x = (["x"], self.x[::2]),
                y = (["y"], self.y[::2]),
                z = (["z"], self.z)
            ),
            data_vars=dict(
                eta=(["y", "x"], eta),
                v=(["y", "x", "z"], v),
                u=(["y", "x", "z"], u)
            ),
            attrs=dict(
                i=n,
                time=n*self.dt
            )
        )

        ds.to_netcdf(self.field_dir+"/"+filename)