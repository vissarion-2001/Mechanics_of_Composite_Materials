import numpy as np
import matplotlib.pyplot as plt 

class Composite_plate:
    def __init__(self):
        
        self.E1 = 39.04e9;
        self.E2 = 14.08e9;
        self.v12 = 0.29
        self.G12 = 4.24e9
        self.tplies = np.ones((12,1))*0.64e-3
        self.angles = [0,0,45,-45,90,90,90,90,-45,45,0,0]    
        self.cov_E1 = 2.64/100
        self.cov_E2 = 2.32/100
        self.cov_v12 = 9.34/100
        self.cov_G12 = 2.34/100
        self.load_coefficient = 3
        self.width = 0.8
        self.length = 1.45
        self.w_dp = 100
        self.l_dp = 100
        self.m = 10
        self.n = 10
        self.Xt = 776.5e6
        self.Xc = 21.82e6
        self.Yt = 53.95e6
        self.Yc = 165e6
        self.S = 56.08e6
        self.cov_Xt = 4.65/100.0
        self.cov_Xc = 3.16/100.0
        self.cov_Yt = 4.78/100.0
        self.cov_Yc = 2.94/100.0
        self.cov_S = 2.0/100.0
        self.ns = 25
        self.gm = 2.2
        self.DT = -60
        self.a1 = 9.17e-6
        self.a2 = 2.35e-5
        
    def complaince_matrix(self):
        S11 = 1/self.E1
        S12 = -self.v12/self.E1
        S22 = 1/self.E2
        S66 = 1/self.G12
        S = np.matrix([[S11, S12, 0],
                      [S12, S22, 0],
                      [0, 0, S66]])
        return S
    
    def construct_h(self):
        t_total = np.sum(self.tplies)
        
        h = np.zeros((len(self.tplies)+1,1))
        h[0] = -t_total/2
        
        for i in range(1, len(self.tplies)+1):
            h[i]=(h[i-1]+self.tplies[i-1])
            
        return np.array(h)
    
    def heights_for_stresses(self):
        hg = self.construct_h()
        hg_repeated = np.repeat(hg[1:-1,0],2).reshape(len(hg[1:-1,0])*2,1)
        return np.vstack((hg[0],hg_repeated,hg[-1]))
        
    
    def convert_deg_to_rad(self):
        angles_rad = []
        for i in self.angles:
            angles_rad.append(i*(np.pi/180))
        return angles_rad
    
    def abd_matrix(self):
        A = np.zeros((3,3))
        B = np.zeros((3,3))
        D = np.zeros((3,3))
        Qfc = np.zeros((3,3, len(self.angles)))
        
        c_angles = self.convert_deg_to_rad()
        
        for j in range(len(self.tplies)):
            a1x = np.cos(-c_angles[j])
            a1y = np.cos(90*(np.pi/180)+c_angles[j])
            a1z = np.cos(90*(np.pi/180))
            a2x = np.cos(90*(np.pi/180)-c_angles[j])
            a2y = np.cos(-c_angles[j])
            a2z = np.cos(90*(np.pi/180))
            a3x = np.cos(90*(np.pi/180))
            a3y = np.cos(90*(np.pi/180))
            a3z = np.cos(0)
            
            g = np.array([[a1x,a1y,a1z], 
                          [a2x, a2y, a2z], 
                          [a3x, a3y, a3z]]).T
            
            Ts = np.matrix([[g[0,0]**2, g[0,1]**2, 2*g[0,0]*g[0,1]],
               [g[1,0]**2, g[1,1]**2, 2*g[1,0]*g[1,1]],
               [g[0,0]*g[1,0], g[0,1]*g[1,1], (g[0,0]*g[1,1] + g[0,1]*g[1,0])]]) 
            
            Te = np.matrix([[g[0,0]**2, g[0,1]**2, g[0,0]*g[0,1]],
               [g[1,0]**2, g[1,1]**2, g[1,0]*g[1,1]],
               [2*g[0,0]*g[1,0], 2*g[0,1]*g[1,1], (g[0,0]*g[1,1] + g[0,1]*g[1,0])]])
            
            
            C = self.complaince_matrix()
            Q = np.linalg.inv(C)
            Qr = np.linalg.inv(Ts)*Q*Te
            
            h = self.construct_h()
            
            A = A + np.multiply(Qr,(h[j+1]-h[j]))
            B = (B + np.multiply(Qr,(h[j+1]**2-h[j]**2)*0.5))
            D = (D + np.multiply(Qr,(h[j+1]**3-h[j]**3)*(1/3)));
            Qfc[:,:,j] = Qr;
            
            ABD_1 = np.hstack((A,B))
            ABD_2 = np.hstack((B,D))
            ABD = np.vstack((ABD_1, ABD_2))
            
        return A, B, D, ABD, Qfc
    
    def disp_pressure_calculations(self):
        x = np.linspace(0, self.width, 100)
        y = np.linspace(0, self.length, 100)
        
        R = self.width/self.length
        
        Dt = self.abd_matrix()[2]
        a = self.width
        b = self.length
        
        w = np.zeros((len(x), len(y)))
        pxy = np.zeros((len(x), len(y)))
        
        
        for k in range(len(x)):
            for l in range(len(y)): 
                w_temp = 0 
                pxy_temp = 0 
                for i in range(1, self.m+1):  
                    for j in range(1, self.n+1):  
                        Dmn = Dt[0,0].item()*i**4+2*(Dt[0,1].item()+2*Dt[2,2].item())*(i**2*j**2*R**2)+Dt[1,1].item()*(j**4*R**4)  
                        pmn = -(4/(a*b))*((-b/(j*np.pi))*((-1)**j-1))*(((-2250*a)/(i*np.pi))*((-1)**i-1)+((450*a*np.pi*(-1)**i)/(np.pi**2*i)))  
                        s1 = np.sin((i*np.pi*x[k])/a)  
                        s2 = np.sin((j*np.pi*y[l])/b)  
                        w_temp = w_temp+((pmn/Dmn)*s1*s2)*(a**4/np.pi**4)*(self.load_coefficient) 
                        pxy_temp = pxy_temp +pmn*s1*s2 
                w[k,l] = w_temp
                pxy[k,l] = pxy_temp
        
        # Calculation of max displacement and point of it
        wmax = np.max(w)
        icoordinates = np.argmax(w)
        
        w_allowable = (1.0/750.0)*self.length
        
        check_disp = np.abs(wmax*1000.0)<=np.abs(w_allowable*1000.0)
        
        if check_disp == True:
            print("Good!")
        else:
            print("Not good!")
        
        return w, pxy, wmax, icoordinates
    
    def moments(self):
        
        x = np.linspace(0, self.width, 100)
        y = np.linspace(0, self.length, 100)
        
        R = self.width/self.length
        
        Dt = self.abd_matrix()[2]
        a = self.width
        b = self.length
        
        Mx = np.zeros((len(x), len(y)))
        My = np.zeros((len(x), len(y)))
        Ms = np.zeros((len(x), len(y)))
        
        for k in range(len(x)):
            for l in range(len(y)): 
                Mx_temp = 0
                My_temp = 0
                Ms_temp = 0
                for i in range(1, self.m+1): 
                    for j in range(1, self.n+1):  
                        Dmn = Dt[0,0].item()*i**4+2*(Dt[0,1].item()+2*Dt[2,2].item())*(i**2*j**2*R**2)+Dt[1,1]*(j**4*R**4) 
                        pmn = -(4/(a*b))*((-b/(j*np.pi))*((-1)**j-1))*(((-2250*a)/(i*np.pi))*((-1)**i-1)+((450*a*np.pi*(-1)**i)/(np.pi**2*i)))   
                        s1 = np.sin((i*np.pi*x[k])/a)    
                        s2 = np.sin((j*np.pi*y[l])/b)  
                        c1 = np.cos((i*np.pi*x[k])/a)
                        c2 = np.cos((j*np.pi*y[l])/b)
                        wmn = (pmn/Dmn)*(a**4/np.pi**4) 
                        Mx_temp = Mx_temp+wmn*s1*s2*(Dt[0,0].item()*(i*np.pi/a)**2+Dt[0,1].item()*(j*np.pi/b)**2)*self.load_coefficient 
                        My_temp = My_temp+wmn*s1*s2*(Dt[0,1].item()*(i*np.pi/a)**2+Dt[1,1].item()*(j*np.pi/b)**2)*self.load_coefficient 
                        Ms_temp = Ms_temp+(-2*Dt[2,2].item())*c1*c2*wmn*((i*j*np.pi**2)/(a*b))*self.load_coefficient
                    Mx[k,l] = Mx_temp
                    My[k,l] = My_temp
                    Ms[k,l] = Ms_temp
        return Mx, My, Ms
    
    def threed_plotting_displacement_pressure(self):
        x = np.linspace(0, self.width, 100)
        y = np.linspace(0, self.length, 100)
        
        w_calc = self.disp_pressure_calculations()[0]
        pxy_calc = self.disp_pressure_calculations()[1]
        
        fig1 =  plt.subplots(figsize=(12,6))
        ax1 = plt.axes(projection="3d")
        ax1.grid()
        X, Y = np.meshgrid(x,y)
        ax1.plot_surface(X,Y,w_calc, cmap="plasma")
        ax1.set_xlabel("a(m)")
        ax1.set_ylabel("b(m)")
        ax1.set_title("Displacement Field over the plate")
        plt.show()
        
        fig2 =  plt.subplots(figsize=(12,6))
        ax2 = plt.axes(projection="3d")
        ax2.grid()
        X, Y = np.meshgrid(x,y)
        ax2.plot_surface(X,Y, pxy_calc, cmap="plasma")
        ax2.set_xlabel("a(m)")
        ax2.set_ylabel("b(m)")
        ax2.set_title("Pressure Field over the plate")
        plt.show()
    
    def plotting_moments(self):
        x = np.linspace(0, self.width, 100)
        y = np.linspace(0, self.length, 100)
        
        Mx_calc = self.moments()[0]
        My_calc = self.moments()[1]
        Ms_calc = self.moments()[2]
        
        fig1 = plt.subplots(figsize=(12,6))
        ax1 = plt.axes(projection="3d")
        ax1.grid()
        X, Y = np.meshgrid(x,y)
        ax1.plot_surface(X,Y,Mx_calc, cmap="plasma")
        ax1.set_xlabel("a(m)")
        ax1.set_ylabel("b(m)")
        ax1.set_title("Mx over the plate")
        plt.show()
        
        fig2 = plt.subplots(figsize=(12,6))
        ax2 = plt.axes(projection="3d")
        ax2.grid()
        X, Y = np.meshgrid(x,y)
        ax2.plot_surface(X,Y,My_calc, cmap="plasma")
        ax2.set_xlabel("a(m)")
        ax2.set_ylabel("b(m)")
        ax2.set_title("My over the plate")
        plt.show()
        
        fig3 = plt.subplots(figsize=(12,6))
        ax3 = plt.axes(projection="3d")
        ax3.grid()
        X, Y = np.meshgrid(x,y)
        ax3.plot_surface(X,Y,Ms_calc, cmap="plasma")
        ax3.set_xlabel("a(m)")
        ax3.set_ylabel("b(m)")
        ax3.set_title("Mσ over the plate")
        plt.show() 
        
    def stresses_calc(self):    
        hs = self.heights_for_stresses() 
        x = np.linspace(0, self.width, 100) 
        y = np.linspace(0, self.length, 100) 
        R = self.width/self.length
        Dt = self.abd_matrix()[2]
        Q = self.abd_matrix()[4]
        a = self.width
        b = self.length
            
        Sx = np.zeros((len(x), len(y), len(hs)))
        Sy = np.zeros((len(x), len(y), len(hs)))
        Ss = np.zeros((len(x), len(y), len(hs)))
            
        lc = 0
        for k in range(len(x)): 
            for l in range(len(y)): 
                lc = 0
                for z in range(0,len(hs),2):
                    Sx_temp1 = 0
                    Sx_temp2 = 0
                    Sy_temp1 = 0
                    Sy_temp2 = 0
                    Ss_temp1 = 0
                    Ss_temp2 = 0 
                    for i in range(1, self.m+1):  
                        for j in range(1, self.n+1):   
                            Dmn = Dt[0,0].item()*i**4+2*(Dt[0,1].item()+2*Dt[2,2].item())*(i**2*j**2*R**2)+Dt[1,1].item()*(j**4*R**4) 
                            pmn = -(4/(a*b))*((-b/(j*np.pi))*((-1)**j-1))*(((-2250*a)/(i*np.pi))*((-1)**i-1)+((450*a*np.pi*(-1)**i)/(np.pi**2*i)))   
                            s1 = np.sin((i*np.pi*x[k])/a)    
                            s2 = np.sin((j*np.pi*y[l])/b)  
                            c1 = np.cos((i*np.pi*x[k])/a)
                            c2 = np.cos((j*np.pi*y[l])/b)
                            Sx_temp1 = Sx_temp1 + (a**2/np.pi**2)*hs[z].item()*(pmn/Dmn)*(Q[0,0,lc].item()*i**2+Q[0,1,lc].item()*j**2*R**2)*s1*s2*self.load_coefficient
                            Sx_temp2 = Sx_temp2 + (a**2/np.pi**2)*hs[z+1].item()*(pmn/Dmn)*(Q[0,0,lc].item()*i**2+Q[0,1,lc].item()*j**2*R**2)*s1*s2*self.load_coefficient
                            Sy_temp1 = Sy_temp1 + (a**2/np.pi**2)*hs[z].item()*(pmn/Dmn)*(Q[0,1,lc].item()*i**2+Q[1,1,lc].item()*j**2*R**2)*s1*s2*self.load_coefficient
                            Sy_temp2 = Sy_temp2 + (a**2/np.pi**2)*hs[z+1].item()*(pmn/Dmn)*(Q[0,1,lc].item()*i**2+Q[1,1,lc].item()*j**2*R**2)*s1*s2*self.load_coefficient
                            Ss_temp1 = Ss_temp1 + (-(2*a**2*R**2)/np.pi**2)*Q[2,2,lc].item()*hs[z].item()*(i*j*pmn/Dmn)*c1*c2*self.load_coefficient
                            Ss_temp2 = Ss_temp2 + (-(2*a**2*R**2)/np.pi**2)*Q[2,2,lc].item()*hs[z+1].item()*(i*j*pmn/Dmn)*c1*c2*self.load_coefficient  
                    Sx[k,l,z] = Sx_temp1
                    Sx[k,l,z+1] = Sx_temp2
                    Sy[k,l,z] = Sy_temp1
                    Sy[k,l,z+1] = Sy_temp2
                    Ss[k,l,z] = Ss_temp1
                    Ss[k,l,z+1] = Ss_temp2
                    lc = lc+1  
        return Sx, Sy, Ss
    
    def q_forces_calculations(self): 
        x = np.linspace(0, self.width, 100)
        y = np.linspace(0, self.length, 100)
        
        R = self.width/self.length
        
        Dt = self.abd_matrix()[2]
        a = self.width
        b = self.length
        
        Qx = np.zeros((len(x), len(y)))
        Qy = np.zeros((len(x), len(y)))
        
        for k in range(len(x)):
            for l in range(len(y)):
                Qx_temp = 0
                Qy_temp = 0
                for i in range(1, self.m+1): 
                    for j in range(1, self.n+1):  
                        Dmn = Dt[0,0].item()*i**4+2*(Dt[0,1].item()+2*Dt[2,2].item())*(i**2*j**2*R**2)+Dt[1,1].item()*(j**4*R**4) 
                        pmn = -(4/(a*b))*((-b/(j*np.pi))*((-1)**j-1))*(((-2250*a)/(i*np.pi))*((-1)**i-1)+((450*a*np.pi*(-1)**i)/(np.pi**2*i)))   
                        s1 = np.sin((i*np.pi*x[k])/a)    
                        s2 = np.sin((j*np.pi*y[l])/b)  
                        c1 = np.cos((i*np.pi*x[k])/a)
                        c2 = np.cos((j*np.pi*y[l])/b)
                        wmn = (pmn/Dmn)*(a**4/np.pi**4)
                        Qx_temp = Qx_temp + wmn*c1*s2*((Dt[0,0].item()*(i*np.pi/a)**2+Dt[0,1].item()*(j*np.pi/b)**2)*(i*np.pi/a)+2*Dt[2,2].item()*((i*j*np.pi**2)/(a*b))*(j*np.pi/b))
                        Qy_temp = Qy_temp + wmn*s1*c2*((Dt[0,1].item()*(i*np.pi/a)**2+Dt[1,1].item()*(j*np.pi/b)**2)*(j*np.pi/a)+2*Dt[2,2].item()*((i*j*np.pi**2)/(a*b))*(i*np.pi/b))
                Qx[k,l] = Qx_temp
                Qy[k,l] = Qy_temp
        return Qx, Qy
    
    def shear_stresses(self):
        h_s = self.heights_for_stresses() 
        x = np.linspace(0, self.width, 100)
        y = np.linspace(0, self.length, 100)
        
        R = self.width/self.length
        
        Dt = self.abd_matrix()[2]
        a = self.width
        b = self.length
        Q = self.abd_matrix()[4]
        
        txy = np.zeros((len(x), len(y), len(h_s)))
        tyz = np.zeros((len(x), len(y), len(h_s)))
        
        for k in range(len(x)):
            for l in range(len(y)):
                lc = 0
                for z in range(0,len(h_s),2):
                    t1_temp = 0
                    t2_temp = 0
                    for i in range(1, self.m+1):
                        for j in range(1, self.n+1):
                            Dmn = Dt[0,0].item()*i**4+2*(Dt[0,1].item()+2*Dt[2,2].item())*(i**2*j**2*R**2)+Dt[1,1].item()*(j**4*R**4) 
                            pmn = -(4/(a*b))*((-b/(j*np.pi))*((-1)**j-1))*(((-2250*a)/(i*np.pi))*((-1)**i-1)+((450*a*np.pi*(-1)**i)/(np.pi**2*i)))   
                            s1 = np.sin((i*np.pi*x[k])/a)    
                            s2 = np.sin((j*np.pi*y[l])/b)  
                            c1 = np.cos((i*np.pi*x[k])/a)
                            c2 = np.cos((j*np.pi*y[l])/b)
                            t1_temp = t1_temp + ((i*pmn)/Dmn)*(Q[0,0,lc].item()*i**2+(Q[0,1,lc].item()+2*Q[2,2,lc].item())*j**2*R**2)*c1*s2*self.load_coefficient
                            t2_temp = t2_temp + ((j*R*pmn)/Dmn)*(Q[1,1,lc].item()*j**2*R**2+(Q[0,1,lc].item()+2*Q[2,2,lc].item())*i**2)*c2*s1*self.load_coefficient
                    txy[k,l,z] = (-a/np.pi)*((h_s[z][0].item()**2-(sum(self.tplies)**2/4))/2)*t1_temp
                    txy[l,l,z+1] = (-a/np.pi)*((h_s[z+1][0].item()**2-(sum(self.tplies)**2/4))/2)*t1_temp
                    tyz[k,l,z] = (-a/np.pi)*((h_s[z][0].item()**2-(sum(self.tplies)**2/4))/2)*t2_temp
                    tyz[k,l,z+1] = (-a/np.pi)*((h_s[z+1][0].item()**2-(sum(self.tplies)**2/4))/2)*t2_temp
                    lc = lc+1 
        return txy, tyz

    def plotting_stresses(self, h_z):
        
        x = np.linspace(0, self.width, 100)
        y = np.linspace(0, self.length, 100)
        
        Sx_calc = self.stresses_calc()[0]
        Sy_calc = self.stresses_calc()[1]
        Ss_calc = self.stresses_calc()[2]
        
        fig1 = plt.subplots(figsize=(12,6))
        ax1 = plt.axes(projection="3d")
        ax1.grid()
        X, Y = np.meshgrid(x,y)
        ax1.plot_surface(X,Y,Sx_calc[:,:,h_z], cmap="plasma")
        ax1.set_xlabel("a(m)")
        ax1.set_ylabel("b(m)")
        ax1.set_title("Sx over the plate")
        plt.show()
        
        fig2 = plt.subplots(figsize=(12,6))
        ax2 = plt.axes(projection="3d")
        ax2.grid()
        X, Y = np.meshgrid(x,y)
        ax2.plot_surface(X,Y,Sy_calc[:,:,h_z], cmap="plasma")
        ax2.set_xlabel("a(m)")
        ax2.set_ylabel("b(m)")
        ax2.set_title("Sy over the plate")
        plt.show()
        
        fig3 = plt.subplots(figsize=(12,6))
        ax3 = plt.axes(projection="3d")
        ax3.grid()
        X, Y = np.meshgrid(x,y)
        ax3.plot_surface(X,Y,Ss_calc[:,:,h_z], cmap="plasma")
        ax3.set_xlabel("a(m)")
        ax3.set_ylabel("b(m)")
        ax3.set_title("Ss over the plate")
        plt.show() 
        
    def plotting_q_forces(self):
        
        x = np.linspace(0, self.width, 100)
        y = np.linspace(0, self.length, 100)
            
        Qx_calc = self.q_forces_calculations()[0]
        Qy_calc = self.q_forces_calculations()[1]
            
        fig1 = plt.subplots(figsize=(12,6))
        ax1 = plt.axes(projection="3d")
        ax1.grid()
        X, Y = np.meshgrid(x,y)
        ax1.plot_surface(X,Y,Qx_calc, cmap="plasma")
        ax1.set_xlabel("a(m)")
        ax1.set_ylabel("b(m)")
        ax1.set_title("Qx over the plate")
        plt.show()
            
        fig2 = plt.subplots(figsize=(12,6))
        ax2 = plt.axes(projection="3d")
        ax2.grid()
        X, Y = np.meshgrid(x,y)
        ax2.plot_surface(X,Y,Qy_calc, cmap="plasma")
        ax2.set_xlabel("a(m)")
        ax2.set_ylabel("b(m)")
        ax2.set_title("Qy over the plate")
        plt.show()
    
    def mechanical_strains(self):
        
        h_s = self.heights_for_stresses() 
        x = np.linspace(0, self.width, 100)
        y = np.linspace(0, self.length, 100)
        
        R = self.width/self.length
        
        Dt = self.abd_matrix()[2]
        a = self.width
        b = self.length
        
        emx = np.zeros((len(x),len(y),len(h_s)))
        emy = np.zeros((len(x),len(y),len(h_s)))
        ems = np.zeros((len(x),len(y),len(h_s)))
        
        for k in range(len(x)):
            for l in range(len(y)):
                emx_temp = np.zeros((1,len(h_s)))
                emy_temp = np.zeros((1,len(h_s)))
                ems_temp = np.zeros((1,len(h_s)))
                kxx = 0
                kyy = 0
                kss = 0
                for i in range(1, self.m+1):
                    for j in range(1, self.n+1):
                        Dmn = Dt[0,0].item()*i**4+2*(Dt[0,1].item()+2*Dt[2,2])*(i**2*j**2*R**2)+Dt[1,1]*(j**4*R**4) 
                        pmn = -(4/(a*b))*((-b/(j*np.pi))*((-1)**j-1))*(((-2250*a)/(i*np.pi))*((-1)**i-1)+((450*a*np.pi*(-1)**i)/(np.pi**2*i)))
                        kxx = kxx + (a**4/np.pi**4)*(pmn/Dmn)*(self.m*np.pi/a)**2*np.sin(self.m*np.pi*x[k]/a)*np.sin(self.n*np.pi*y[l]/b)*self.load_coefficient
                        kyy = kyy + (a**4/np.pi**4)*(pmn/Dmn)*(self.n*np.pi/b)**2*np.sin(self.m*np.pi*x[k]/a)*np.sin(self.n*np.pi*y[l]/b)*self.load_coefficient
                        kss = kss -2*(a**4/np.pi**4)*(pmn/Dmn)*(self.m*self.n*np.pi**2/(a*b))*np.cos(self.m*np.pi*x[k]/a)*np.cos(self.n*np.pi*y[l]/b)*self.load_coefficient
                emx_temp = kxx*h_s
                emy_temp = kyy*h_s
                ems_temp = kss*h_s
                emx[k,l,:] = emx_temp.flatten()
                emy[k,l,:] = emy_temp.flatten()
                ems[k,l,:] = ems_temp.flatten()
        return emx, emy, ems
    
    def plotting_strains(self, x_c, y_c):
        strains = self.mechanical_strains()
        z_all = self.heights_for_stresses()
        fig,ax = plt.subplots(1,3, figsize=(12,6))
        ax[0].plot(strains[0][x_c,y_c,:], z_all)
        ax[0].set_xlabel("εx")
        ax[0].set_ylabel("Z")
        ax[1].plot(strains[1][x_c,y_c,:], z_all)
        ax[1].set_xlabel("εy")
        ax[1].set_ylabel("Z")
        ax[2].plot(strains[2][x_c,y_c,:], z_all)
        ax[2].set_xlabel("εs")
        ax[2].set_ylabel("Z")
        plt.tight_layout()
    
    def R_values(self):
        R_xt = (self.Xt*(1-self.cov_Xt*(1.645+1.645/np.sqrt(self.ns))))/self.gm
        R_xc = (self.Xc*(1-self.cov_Xc*(1.645+1.645/np.sqrt(self.ns))))/self.gm 
        R_yt = (self.Yt*(1-self.cov_Yt*(1.645+1.645/np.sqrt(self.ns))))/self.gm 
        R_yc = (self.Yc*(1-self.cov_Yc*(1.645+1.645/np.sqrt(self.ns))))/self.gm 
        R_s = (self.S*(1-self.cov_S*(1.645+1.645/np.sqrt(self.ns))))/self.gm
        return R_xt, R_xc, R_yt, R_yc, R_s
    
    def tpc(self, stresses):
        
        h_s = self.heights_for_stresses() 
        
        x = np.linspace(0, self.width, 100)
        y = np.linspace(0, self.length, 100)
        
        Sx_physical = np.zeros((len(x), len(y), len(h_s)))
        Sy_physical = np.zeros((len(x), len(y), len(h_s)))
        Ss_physical = np.zeros((len(x), len(y), len(h_s)))
        
        for k in range(len(x)):
            for l in range(len(y)):
                angle_counter = 0
                for z in range(0,len(h_s),2):
                    mc = np.cos(self.convert_deg_to_rad()[angle_counter])
                    ms = np.sin(self.convert_deg_to_rad()[angle_counter])
                    Ttr =  np.array([[mc**2, ms**2, 2*ms*mc], 
                                     [ms**2, mc**2, -2*ms*mc], 
                                     [-ms*mc, ms*mc, (mc**2-ms**2)]])
                    St1 = np.array([stresses[0][k,l,z], stresses[1][k,l,z], stresses[2][k,l,z]]).reshape((3,1))
                    St2 = np.array([stresses[0][k,l,z+1], stresses[1][k,l,z+1], stresses[2][k,l,z+1]]).reshape((3,1))
                    
                    s_phys_1 = np.dot(Ttr,St1) 
                    s_phys_2 = np.dot(Ttr,St2)
                    
                    Sx_physical[k,l,z] = s_phys_1[0].item()
                    Sx_physical[k,l,z+1] = s_phys_2[0].item()
                    Sy_physical[k,l,z] = s_phys_1[1].item()
                    Sy_physical[k,l,z+1] = s_phys_2[1].item()
                    Ss_physical[k,l,z] = s_phys_1[2].item()
                    Ss_physical[k,l,z+1] =s_phys_2[2].item()
                    angle_counter +=1
        return Sx_physical, Sy_physical, Ss_physical
    
    def tsai_wu_fpf(self, stresses):
        
        h_s = self.heights_for_stresses() 
        
        x = np.linspace(0, self.width, 100)
        y = np.linspace(0, self.length, 100)
        
        fe_all = np.zeros((len(x), len(y), len(h_s)))
        
        for k in range(len(x)):
            for l in range(len(y)):
                for z in range(0, len(h_s), 2):
                    
                    F1 = 1/self.R_values()[0]-1/self.R_values()[1] 
                    F2 = 1/self.R_values()[2]-1/self.R_values()[3] 
                    F11 = 1/(self.R_values()[0]*self.R_values()[1]) 
                    F22 = 1/(self.R_values()[2]*self.R_values()[3]) 
                    F12 = -0.5*(F11+F22);
                    F66 = 1/self.R_values()[4]
                    
                    b1 = F1*self.tpc(stresses)[0][k,l,z]+F2*self.tpc(stresses)[1][k,l,z]
                    b2 = F1*self.tpc(stresses)[0][k,l,z+1]+F2*self.tpc(stresses)[1][k,l,z+1]
                    
                    
                    c1 = F11*self.tpc(stresses)[0][k,l,z]**2 + F22*self.tpc(stresses)[1][k,l,z]**2 + F66*self.tpc(stresses)[2][k,l,z]**2 + 2*F12*self.tpc(stresses)[0][k,l,z]*self.tpc(stresses)[1][k,l,z]
                    c2 = F11*self.tpc(stresses)[0][k,l,z+1]**2 + F22*self.tpc(stresses)[1][k,l,z+1]**2 + F66*self.tpc(stresses)[2][k,l,z]**2 + 2*F12*self.tpc(stresses)[0][k,l,z+1]*self.tpc(stresses)[1][k,l,z+1]
                    
                    fe_all[k,l,z] = (b1+np.sqrt(b1**2+4*c1))/2
                    fe_all[k,l,z+1] = (b2+np.sqrt(b2**2+4*c2))/2
        
        L_tsai_Wu = 1/np.max(fe_all[:])-1
                    
        return fe_all, np.max(fe_all), L_tsai_Wu
    
    def max_criterion_failure(self, stresses):
        
        Sx_p, Sy_p, Ss_s = self.tpc(stresses)
        
        fs1_p = np.abs(np.max(Sx_p))/self.R_values()[0] 
        fs1_n = np.abs(np.min(Sx_p))/self.R_values()[1] 
        fs2_p = np.abs(np.max(Sy_p))/self.R_values()[2] 
        fs2_n = np.abs(np.min(Sy_p))/self.R_values()[3] 
        fs6_p = np.abs(np.max(Ss_s))/self.R_values()[4] 
        fs6_n = np.abs(np.min(Ss_s))/self.R_values()[4] 
        fs_All = [fs1_p,fs1_n,fs2_p,fs2_n,fs6_p,fs6_n];
        
        return max(fs_All)
    
    def safety_of_margin_calculations(self, stresses):
        
        min_s1p = np.min(self.tpc(stresses)[0])
        max_s1p = np.max(self.tpc(stresses)[0])
        min_s2p = np.min(self.tpc(stresses)[1])
        max_s2p = np.max(self.tpc(stresses)[1])
        min_s6p = np.min(self.tpc(stresses)[2])
        max_s6p = np.max(self.tpc(stresses)[2]) 
        
        L1 = np.abs(self.R_values()[0]/max_s1p)-1
        L2 = np.abs(self.R_values()[1]/min_s1p)-1
        L3 = np.abs(self.R_values()[2]/max_s2p)-1
        L4 = np.abs(self.R_values()[3]/min_s2p)-1
        L5 = np.abs(self.R_values()[4]/max_s6p)-1
        L6 = np.abs(self.R_values()[5]/min_s6p)-1 
        
        L = np.max([L1, L2, L3, L4, L5, L6])
        
        return np.min(L)
    
    def thermal_expansion_calculator(self):
        a = np.array([[self.a1],
                      [self.a2],
                      [0]])
        
        a_b1 = np.zeros((3,1))
        a_b2 = np.zeros((3,1))
        
        h = self.heights_for_stresses()
        
        c_angles = self.convert_deg_to_rad()
        afc = np.zeros((3,1,len(self.tplies)))
        
        Qfc = self.abd_matrix()[4]
        ABD = self.abd_matrix()[3]
        
        for i in range(len(self.tplies)):
            c = np.cos(c_angles[i])
            s = np.sin(c_angles[i]) 
            
            Tn = np.array([[c**2,s**2,2*s*c],
                       [s**2,c**2,-2*s*c],
                       [-s*c,s*c,c**2-s**2]])
            
            a_all = np.dot(Tn,a)
            afc[:,:,i] = a_all
            
            a_b1 = (a_b1 + Qfc[:,:,i]*a_all*(h[i+1]-h[i]))
            a_b2 = (a_b2 + Qfc[:,:,i]*a_all*(h[i+1]**2-h[i]**2)*0.5)
            
            
        af =  np.linalg.inv(ABD[0:3,0:3])*a_b1 + np.linalg.inv(ABD[0:3,3:])*a_b2
        
        return af, afc
    
    def fictious_forces_and_moments(self):
        
        h = self.heights_for_stresses()
        
        N1 = 0
        M1 = 0
        
        Q_fc = self.abd_matrix()[4]
        
        afc = self.thermal_expansion_calculator()
        
        for i in range(len(self.tplies)):
            N1 = N1 + (self.DT*Q_fc[:,:,i]*afc[:,:,i]*(h[i+1]-h[i])) 
            M1 = (M1 + (self.DT*Q_fc[:,:,i]*afc[:,:,i]*(h[i+1]**2-h[i]**2)*0.5))
            
    
    def thermal_strain_calculations(self):
        
        h_s = self.heights_for_stresses() 
        
        x = np.linspace(0, self.width, 100)
        y = np.linspace(0, self.length, 100)
        
        etx = np.zeros((len(x),len(y),len(h_s)))
        ety = np.zeros((len(x),len(y),len(h_s)))
        etz = np.zeros((len(x),len(y),len(h_s)))
        
        Mx = self.moments()[0]
        My = self.moments()[1]
        Ms = self.moments()[2]
        
        ABD = self.abd_matrix()[3]
        
        for k in range(len(x)):
            for l in range(len(y)):
                eti = np.linalg.inv(ABD[0:3,3:6])*np.array([[Mx[k,l]],
                                                            [My[k,l]],
                                                            [Ms[k,l]]])
                kti = np.linalg.inv(ABD[3:6,3:6])*np.array([[Mx[k,l]],
                                                            [My[k,l]],
                                                            [Ms[k,l]]])
                
                etx[k,l,:] = np.array(eti[0] + h_s*kti[0]).flatten()
                ety[k,l,:] = np.array(eti[1] + h_s*kti[1]).flatten()
                etz[k,l,:] = np.array(eti[2] + h_s*kti[2]).flatten()
        
        return etx, ety, etz
    
    def free_thermal_strains(self):
        
        h_s = self.heights_for_stresses() 
        
        x = np.linspace(0, self.width, 100)
        y = np.linspace(0, self.length, 100)
        
        etx_f = np.zeros((len(x),len(y),len(h_s)))
        ety_f = np.zeros((len(x),len(y),len(h_s)))
        etz_f = np.zeros((len(x),len(y),len(h_s)))
        
        afc = self.thermal_expansion_calculator()[1]
        
        for k in range(len(x)):
            for l in range(len(y)):
                c = 0
                for z in range(0, len(h_s), 2):
                    etf = afc[:,:,c]*self.DT
                    etx_f[z] = etf[0]
                    etx_f[z+1] = etf[0]
                    ety_f[z] = etf[0]
                    ety_f[z+1] = etf[0]
                    etz_f[z] = etf[0]
                    etz_f[z+1] = etf[0]
                    c +=1
        return etx_f, ety_f, etz_f
                    
    def thermal_stress_calculations(self):
        
        h_s = self.heights_for_stresses() 
        
        x = np.linspace(0, self.width, 100)
        y = np.linspace(0, self.length, 100)
        
        Qfc = self.abd_matrix()[4]
        
        sxt = np.zeros((len(x), len(y), len(h_s)))
        syt = np.zeros((len(x), len(y), len(h_s)))
        sst = np.zeros((len(x), len(y), len(h_s)))
        
        diff_ex = self.thermal_strain_calculations()[0] - self.free_thermal_strains()[0]
        diff_ey = self.thermal_strain_calculations()[1] - self.free_thermal_strains()[1]
        diff_es = self.thermal_strain_calculations()[2] - self.free_thermal_strains()[2]
        
        sx = self.stresses_calc()[0]
        sy = self.stresses_calc()[1]   
        ss = self.stresses_calc()[2]
        
        for k in range(len(x)):
            for l in range(len(y)): 
                qa = 0
                for z in range(0, len(h_s), 2):
                    sxt[k,l,z] = sx[k,l,z].item() + np.dot(Qfc[0,:,qa].reshape(1,3),np.array([[diff_ex[k,l,z].item()],[diff_ey[k,l,z].item()],[diff_es[k,l,z].item()]]))
                    sxt[k,l,z+1] = sx[k,l,z+1].item() + np.dot(Qfc[0,:,qa].reshape(1,3),np.array([[diff_ex[k,l,z+1].item()],[diff_ey[k,l,z+1].item()],[diff_es[k,l,z+1].item()]]))
                    syt[k,l,z] = sy[k,l,z].item() + np.dot(Qfc[1,:,qa].reshape(1,3),np.array([[diff_ex[k,l,z].item()],[diff_ey[k,l,z].item()],[diff_es[k,l,z].item()]]))
                    syt[k,l,z+1] = sy[k,l,z+1].item() + np.dot(Qfc[1,:,qa].reshape(1,3),np.array([[diff_ex[k,l,z+1].item()],[diff_ey[k,l,z+1].item()],[diff_es[k,l,z+1].item()]]))
                    sst[k,l,z] = ss[k,l,z].item() + np.dot(Qfc[2,:,qa].reshape(1,3),np.array([[diff_ex[k,l,z].item()],[diff_ey[k,l,z].item()],[diff_es[k,l,z].item()]]))
                    sst[k,l,z+1] = ss[k,l,z+1].item() + np.dot(Qfc[2,:,qa].reshape(1,3),np.array([[diff_ex[k,l,z+1].item()],[diff_ey[k,l,z+1].item()],[diff_es[k,l,z+1].item()]]))
                    qa = qa+1 
                    
        return sxt, syt, sst
                    

if __name__ == "__main__": 
    material = Composite_plate()  
    complaince_matrix = material.complaince_matrix()
    heights = material.construct_h()
    heights_for_stresses = material.heights_for_stresses()
    angles = material.convert_deg_to_rad()
    A = material.abd_matrix()[0]
    B = material.abd_matrix()[1]
    D = material.abd_matrix()[2]
    ABD = material.abd_matrix()[3]
    Q_per_lamina = material.abd_matrix()[4]
    moments = material.moments()
    d = material.disp_pressure_calculations()
    material.threed_plotting_displacement_pressure()
    stresses = material.stresses_calc()
    stresses_t = material.thermal_stress_calculations()
    q = material.q_forces_calculations
    material.plotting_moments()
    material.plotting_stresses(5)
    material.plotting_q_forces()
    s = material.shear_stresses()
    material.plotting_strains(50, 50)
    material.tsai_wu_fpf(stresses)
    fs_all_mechanical = material.max_criterion_failure(stresses)
    fs_all_thermal = material.max_criterion_failure(stresses_t)
    sft_m = material.safety_of_margin_calculations(stresses)
    sft_t = material.safety_of_margin_calculations(stresses_t)
    
    
    
    
    
    
                           
        
        
    

        
        
        
        

        
        
