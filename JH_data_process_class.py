#########################################
# Data Process
# JH4209@KrappLab
# 2017-12-06
#########################################

import numpy
from scipy import signal

import scipy.io as sio
import matplotlib.pyplot as plt #as plt for convenience
from matplotlib import gridspec
import sys,os
# import time
from scipy import interpolate


class JH_swing_rig_data_process():

	def __init__(self,matfolder,threshold=2.8):
		self.matfolder = os.path.join(matfolder, '') #... add trailing slash to folder string
		# self.matfile=''	

		self.anglist=[-1,90,180,270,0,45,135,225,315]

		self.fs = 20000
		self.duration = 14
		self.datalen = self.fs * self.duration
		self.t = numpy.arange(start=0, stop=self.duration, step=1.0/self.fs, dtype=numpy.float64)
	
		self.threshold=threshold		#threshold for the spike
		self.stim_threshold=1.0 	#threshold for the sync

#########################################
############# Basic functions ###########
#########################################
	
	def load_raw(self,matfile=''):
		# print ('Plotting ' + matfile + ' ...')
		matdata = sio.loadmat(matfile)
		data = matdata.get('data'); del matdata;
		data = numpy.transpose(data)
		# t = numpy.arange(start=0, stop=duration, step=1.0/fs,dtype=numpy.float64)
		stim=data[1]
		resp=data[0]
		return stim,resp



	def stim2sync(self, stim):
		sync=stim>self.stim_threshold
		sync = numpy.zeros((self.datalen))
		sync_rise=0
		sync_fall=0
		for i in range(1,self.datalen):
			if (stim[i-1]<1.0 and stim[i] >= 1.0):
				sync_rise = i
			if (stim[i-1]>1.0 and stim[i] <= 1.0):
				sync_fall = i
			if (sync_rise<sync_fall and (sync_fall-sync_rise)>(self.fs/2)):
				sync[sync_rise:sync_fall] = 1
		return sync



	def raw2raster(self, raw):
		# threshold = 2.5 + 0.3
		raster = numpy.zeros(self.datalen, dtype=numpy.uint)
		for i in range(1,self.datalen):
			if (raw[i]>=self.threshold and raw[i-1]<=self.threshold):
				raster[i] = 1
		return raster



	def raster2rate(self, raster):
		prev_idx = 0
		ISI_rate = numpy.zeros(self.datalen, dtype=numpy.uint)
		for i in range(1,self.datalen):
			if (raster[i] == 1):
				temp_rate = self.fs/(i - prev_idx)
				for j in range(prev_idx, i):
					ISI_rate[j] = temp_rate
				prev_idx = i
		return ISI_rate



	def plot(self,filename,stim,sync,resp,raster,rate):
		fig1 = plt.figure(num=None, figsize=(8, 6), dpi=150, facecolor='w', edgecolor='k')
		gs = gridspec.GridSpec(3, 1)#, width_ratios=[4,1])
	#--------------------- Subplot 1: Photodiode Binary Signal -------------------
		ax0=plt.subplot(gs[0])
		plt.plot(self.t,stim,'r')
		plt.plot(self.t,sync,'g')
		plt.ylabel("Stim (V)")
		plt.text(0*1.5+1,0,"null",horizontalalignment='center')
		for i in range(1,len(self.anglist)):
			plt.text(i*1.5+1,0,self.anglist[i],horizontalalignment='center')
		plt.title(filename)
	#------------------------ Subplot 3: Action Potentials -----------------------
		ax1=plt.subplot(gs[1], sharex=ax0)
		plt.plot(self.t,resp,'b')
		plt.axhspan(2.5,self.threshold,0, self.duration, color='m', alpha=0.5)
		plt.ylabel("Resp (V)")
	#---------------------------- Subplot 5: Spike Train -------------------------
		plt.subplot(gs[2], sharex=ax0)
		plt.plot(self.t,raster*400,'r')
		plt.plot(self.t,rate,'m')
		plt.axis([0, self.duration, 0, 400])
		plt.ylabel("ISI rate (Hz)")
		plt.xlabel("time (s)")

		plt.show()

	

#########################################
########### adv function ################
#########################################
	def raw2raster_dVttdt(self,data,plotflag=0):
		# plotflag=0
		dv_threshold = 0.2

		if plotflag==1:
			plt.figure()

		raster1 = numpy.zeros(self.datalen, dtype=numpy.uint)
		raster2 = numpy.zeros(self.datalen, dtype=numpy.uint)

		for i in range(1,self.datalen):	#...spike sorting based on dv/dt
			if ( data[i-1]<=self.threshold<data[i]):
				j=i
				while(data[j-1]<data[j]):
					if j<=0:
						break			
					j-=1
				PreTroughIndex=j
				j=i
				while(data[j-1]<data[j]):
					if j>=self.datalen-1:
						break
					j+=1
				PeakIndex=j
				while(data[j-1]>data[j]):
					if j>=self.datalen-1:
						break
					j+=1
				PostTroughIndex=j

				dv=data[PreTroughIndex]-data[PostTroughIndex]
				dt=PostTroughIndex - PreTroughIndex

				if plotflag==1:
					plt.plot(dt,dv,'ro',mfc='none')

				if dv > dv_threshold:
					raster1[PeakIndex] = 1
				else:
					raster2[PeakIndex] = 1

		if plotflag==1:
			plt.plot([5,30],[0.2,0.2],'g')
			plt.show()

		return raster1,raster2


	def raw2raster_dVptdt(self,data,plotflag=0):
		# plotflag=0
		dv_threshold = 0.2

		if plotflag==1:
			plt.figure()

		raster1 = numpy.zeros(self.datalen, dtype=numpy.uint)
		raster2 = numpy.zeros(self.datalen, dtype=numpy.uint)

		for i in range(1,self.datalen):	#...spike sorting based on dv/dt
			if ( data[i-1]<=self.threshold<data[i]):
				# j=i
				# while(data[j-1]<data[j]):
				# 	if j<=0:
				# 		break			
				# 	j-=1
				# PreTroughIndex=j
				j=i
				while(data[j-1]<data[j]):
					if j>=self.datalen-1:
						break
					j+=1
				PeakIndex=j
				while(data[j-1]>data[j]):
					if j>=self.datalen-1:
						break
					j+=1
				PostTroughIndex=j

				dv=data[PeakIndex]-data[PostTroughIndex]
				dt=PostTroughIndex - PeakIndex



				if ((dv-1.3)/dt > (0.45-1.3)/12):
					if plotflag==1:
						plt.plot(dt,dv,'bo',mfc='none')
					raster1[PeakIndex] = 1
				else:
					if plotflag==1:
						plt.plot(dt,dv,'ro',mfc='none')
					raster2[PeakIndex] = 1

		if plotflag==1:
			plt.plot([0,12],[1.3,0.45],'g')
			plt.xlabel('dt')
			plt.ylabel('dv')
			plt.title('Spike clustering')
			plt.show()

		return raster1,raster2

#########################################
############# get function ##############
#########################################


	def getXY(self,filename):
		x=filename[filename.find('[A')+2:filename.find('E')]
		y=filename[filename.find('E')+1:filename.find('][SN')]
		# print('A=',x,'E=',y)
		x=int(float(x))
		y=int(float(y))
		# print(x,y)
		return x,y


	def getUV(self,sync,raster):
		LMS=numpy.zeros(9)
		LMD=self.anglist
		LMS_index=0
		for i in range(1,self.datalen):
			if sync[i]==1 and raster[i]==1:
				LMS[LMS_index]+=1
			if sync[i-1]==1 and sync[i]==0:
				LMS_index+=1
		# print(LMS,LMD)
		u=0
		v=0
		for j in range(1,len(LMS)):
			u+=LMS[j] * numpy.cos(LMD[j]*numpy.pi/180.0)
			v+=LMS[j] * numpy.sin(LMD[j]*numpy.pi/180.0)
		# 	print(u,v)
		# print('-----------')	
		# print(u,v)	
		# print(numpy.sqrt(u**2+v**2),numpy.arctan2(v,u)*180.0/numpy.pi)
		return u,v






	def getAllXYUV2(self, log=0):

		matfilelist=[]
		for (dirpath,dirnames,filenames) in os.walk(self.matfolder):
			matfilelist.extend(filenames)
			break

		XYUV_ipsi=numpy.zeros((0,4))
		XYUV_contra=numpy.zeros((0,4))

		for i in range(len(matfilelist)):
			print('Progress:' ,i+1,'/',len(matfilelist))

			matfile=self.matfolder+matfilelist[i]

			stim,resp=self.load_raw(matfile)
			sync=self.stim2sync(stim)	
			raster1,raster2=self.raw2raster_dVptdt(resp,plotflag=0)
			# rate=raster2rate(raster1)
			u1,v1=self.getUV(sync,raster1)
			u2,v2=self.getUV(sync,raster2)
			x,y=self.getXY(matfile)

			print(x,y,u1,v1)
			print(x,y,u2,v2)
			XYUV_ipsi = numpy.vstack((XYUV_ipsi,[x,y,u1,v1]))
			XYUV_contra = numpy.vstack((XYUV_contra,[x,y,u2,v2]))

		if not os.path.exists(self.matfolder+"\\result\\"):
		    os.makedirs(self.matfolder+"\\result\\")
		if log==1:
			print('Saving data...')
			sio.savemat(self.matfolder+"result\\XYUV2.mat",{'XYUV_ipsi':XYUV_ipsi,'XYUV_contra':XYUV_contra})
			print('Data saved !')




	def getAllXYUV(self, log=1):

		matfilelist=[]
		for (dirpath,dirnames,filenames) in os.walk(self.matfolder):
			matfilelist.extend(filenames)
			break

		if not os.path.isfile(self.matfolder+"result\\raster.mat"):
			print("Error: need raster.mat ...")
			return 0
		raster_array=sio.loadmat(self.matfolder+'result\\raster.mat')['raster_array']
		sync_array=sio.loadmat(self.matfolder+'result\\raster.mat')['sync_array']
		# print(numpy.shape(raster_array),numpy.shape(sync_array))


		XYUV=numpy.zeros((0,4))

		for i in range(len(matfilelist)):
			print('Progress:' ,i+1,'/',len(matfilelist))

			# print(matfilelist[i])
			matfile=self.matfolder+matfilelist[i]

			x,y=self.getXY(matfile)
			# stim,resp=self.load_raw(matfile)
			# sync=self.stim2sync(stim)	
			# raster=self.raw2raster(resp)
			# rate=raster2rate(raster1)
			u,v=self.getUV(sync_array[i],raster_array[i])
			# u2,v2=getUV(sync,raster2)

			print(x,y,u,v)
			# print(x,y,u2,v2)
			XYUV = numpy.vstack((XYUV,[x,y,u,v]))
			# XYUV_contra = numpy.vstack((XYUV_contra,[x,y,u2,v2]))
		if not os.path.exists(self.matfolder+"\\result\\"):
		    os.makedirs(self.matfolder+"\\result\\")
		if log==1:
			print('Saving data...')
			sio.savemat(self.matfolder+"result\\XYUV.mat",{'XYUV':XYUV})#,'XYUV_contra':XYUV_contra})
			print('Data saved !')



	def getAllRaster_dvdt(self, log=1):
		matfilelist=[]

		for (dirpath,dirnames,filenames) in os.walk(self.matfolder):
			matfilelist.extend(filenames)
			break

		raster_ipsi=numpy.zeros((0,self.datalen))
		raster_contra=numpy.zeros((0,self.datalen))
		sync_array=numpy.zeros((0,self.datalen))

		for i in range(len(matfilelist)):
			print('Progress:' ,i+1,'/',len(matfilelist))

			matfile=self.matfolder+matfilelist[i]

			stim,resp=self.load_raw(matfile)
			sync=self.stim2sync(stim)	
			raster1,raster2=self.raw2raster_dVptdt(resp)
			# u1,v1=self.getUV(sync,raster1)
			# u2,v2=self.getUV(sync,raster2)

			raster_ipsi = numpy.vstack((raster_ipsi,raster1))
			raster_contra = numpy.vstack((raster_contra,raster2))
			sync_array=numpy.vstack((sync_array,sync))

		if not os.path.exists(self.matfolder+"result\\"):
		    os.makedirs(self.matfolder+"result\\")
		if log==1:
			print('Saving data...')
			sio.savemat(self.matfolder+"result\\raster.mat",
									{'raster_ipsi':raster_ipsi,
									'raster_contra':raster_contra,
									'sync_array':sync_array})
			print('Done!')


	def getAllRaster_threshold(self, log=1):
		matfilelist=[]

		for (dirpath,dirnames,filenames) in os.walk(self.matfolder):
			matfilelist.extend(filenames)
			break

		raster_array=numpy.zeros((0,self.datalen))
		sync_array=numpy.zeros((0,self.datalen))

		for i in range(len(matfilelist)):
			print('Progress:' ,i+1,'/',len(matfilelist))

			matfile=self.matfolder+matfilelist[i]

			stim,resp=self.load_raw(matfile)
			sync=self.stim2sync(stim)	
			raster=self.raw2raster(resp)
			# u,v=self.getUV(sync,raster)

			raster_array = numpy.vstack((raster_array,raster))
			sync_array=numpy.vstack((sync_array,sync))


		if log==1:
			if not os.path.exists(self.matfolder+"result\\"):
				os.makedirs(self.matfolder+"result\\")			
			print('Saving data...')
			sio.savemat(self.matfolder+"result\\raster.mat",
									{'raster_array':raster_array.astype(bool),
									'sync_array':sync_array.astype(bool)})
			print('Done!')



#########################################
############# plot function #############
#########################################

	def plotReceptiveField2(self,quiver_scale=1):
		X,Y = numpy.meshgrid(numpy.arange(-180,181,15),numpy.arange(-75,76,15))
		U1 = numpy.zeros_like(X,dtype=numpy.float64)
		V1 = numpy.zeros_like(Y,dtype=numpy.float64)
		U2 = numpy.zeros_like(X,dtype=numpy.float64)
		V2 = numpy.zeros_like(Y,dtype=numpy.float64)

		XYUV_ipsi=sio.loadmat(self.matfolder+'result\\XYUV2.mat')['XYUV_ipsi']
		XYUV_contra=sio.loadmat(self.matfolder+'result\\XYUV2.mat')['XYUV_contra']

		for i in range(len(XYUV_ipsi)):
			x,y,u1,v1=XYUV_ipsi[i]
			Y_index = numpy.where(X[0,:]==x)
			X_index = numpy.where(Y[:,0]==y)
			U1[X_index,Y_index]=-u1 	#...here to compensate up-side-down mount of the fly
			V1[X_index,Y_index]=-v1

			x,y,u2,v2=XYUV_contra[i]
			Y_index = numpy.where(X[0,:]==x)
			X_index = numpy.where(Y[:,0]==y)
			U2[X_index,Y_index]=-u2 	#...here to compensate up-side-down mount of the fly
			V2[X_index,Y_index]=-v2


		plt.figure(num=None, figsize=(16, 9), dpi=80, facecolor='w', edgecolor='k')

		plt.subplot(211)
		plt.quiver(X,Y,U1,V1,scale=quiver_scale,units='xy',color='b')
		plt.title("ipsi-H1-cell mapping (no interpolation)")
		plt.xlabel("Azimuth (deg)")
		plt.ylabel("Elevation (deg)")
		plt.xticks(numpy.arange(-180,181,15))
		plt.yticks(numpy.arange(-75,76,15))

		plt.subplot(212)
		plt.quiver(X,Y,U2,V2,scale=10,units='xy',color='r')
		plt.title("contra-H1-cell mapping (no interpolation)")
		plt.xlabel("Azimuth (deg)")
		plt.ylabel("Elevation (deg)")
		plt.xticks(numpy.arange(-180,181,15))
		plt.yticks(numpy.arange(-75,76,15))

		plt.show()





	def plotReceptiveField(self,quiver_scale=1,interp=1, log=1):
		X,Y = numpy.meshgrid(numpy.arange(-180,181,15),numpy.arange(-75,76,15))
		U = numpy.zeros_like(X,dtype=numpy.float64)
		V = numpy.zeros_like(Y,dtype=numpy.float64)

		if not os.path.isfile(self.matfolder+"result\\XYUV.mat"):
			print("Error: need XYUV.mat ...")
			return 0
		XYUV=sio.loadmat(self.matfolder+'result\\XYUV.mat')['XYUV']

		plt.figure(num=None, figsize=(16, 9), dpi=80, facecolor='w', edgecolor='k')

		for i in range(len(XYUV)):
			x,y,u,v=XYUV[i]
			Y_index = numpy.where(X[0,:]==x)
			X_index = numpy.where(Y[:,0]==y)

			U[X_index,Y_index]=-u 	#...here to compensate up-side-down mount of the fly
			V[X_index,Y_index]=-v

		# print('len of XYUV is: ', len(XYUV))
		# print(XYUV[:,0],XYUV[:,1],XYUV[:,2],XYUV[:,3])
		if interp==1:
		# 	fU = interpolate.interp2d(XYUV[:,0],XYUV[:,1],-XYUV[:,2],kind='linear')
		# 	fV = interpolate.interp2d(XYUV[:,0],XYUV[:,1],-XYUV[:,3],kind='linear')
		# 	iU = fU(numpy.arange(-180,181,15),numpy.arange(-75,76,15))
		# 	iV = fV(numpy.arange(-180,181,15),numpy.arange(-75,76,15))
			iU = interpolate.griddata((XYUV[:,0],XYUV[:,1]),-XYUV[:,2],(X,Y),method='linear')
			iV = interpolate.griddata((XYUV[:,0],XYUV[:,1]),-XYUV[:,3],(X,Y),method='linear')

			plt.quiver(X,Y,iU,iV,scale=quiver_scale,units='xy',color='r')
			plt.title(self.matfolder.split('\\')[-2])
		else:
			plt.title(self.matfolder.split('\\')[-2]+" (no interpolation)")


		plt.quiver(X,Y,U,V,scale=quiver_scale,units='xy',color='k')
		# plt.title(self.matfolder)
		plt.xlabel("Azimuth (deg)")
		plt.ylabel("Elevation (deg)")
		plt.xticks(numpy.arange(-180,181,15))
		plt.yticks(numpy.arange(-75,76,15))

		if log==1 :
			if interp==1:
				plt.savefig(self.matfolder + 'result\\map_interp.png')
			else:
				plt.savefig(self.matfolder + 'result\\map.png')

		
		plt.show()



	def plotRaster2(self):
		raster_ipsi=sio.loadmat(self.matfolder+'result\\raster.mat')['raster_ipsi']
		raster_contra=sio.loadmat(self.matfolder+'result\\raster.mat')['raster_contra']
		sync_array=sio.loadmat(self.matfolder+'result\\raster.mat')['sync_array']
		print(numpy.shape(raster_ipsi),numpy.shape(raster_contra),numpy.shape(sync_array))

		plt.figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')

		for i in range(21):
			ipsi=raster_ipsi[i]
			contra=raster_contra[i]
			ipsi[ipsi==0]=numpy.nan
			contra[contra==0]=numpy.nan
			plt.plot(self.t,ipsi*0.25+i,'b|')
			plt.plot(self.t,contra*0.5+i,'r|')
			plt.plot(self.t,sync_array[i]*0.75+i,'g')
			plt.title("raster data from 0 to 20")
			plt.xlabel('Time (s)')
			plt.ylabel('Sorted spikes')
		plt.show()




	def plotRaster(self, log =1):
		raster_array=sio.loadmat(self.matfolder+'result\\raster.mat')['raster_array']
		sync_array=sio.loadmat(self.matfolder+'result\\raster.mat')['sync_array']
		print(numpy.shape(raster_array),numpy.shape(sync_array))

		plt.figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')

		for i in range(84):
			raster=raster_array[i].astype(numpy.float64)
			raster[raster==0]=numpy.nan
			plt.plot(self.t,raster*0.33+i,'b|',lw=0.5)
			plt.plot(self.t,sync_array[i]*0.66+i,'g')
			plt.title(self.matfolder.split('\\')[-2])
			plt.xlabel('Time (s)')
			plt.ylabel('Sorted spikes')

		if log==1 :
			plt.savefig(self.matfolder + 'result\\raster.png')
		plt.show()





	def plotSNR(self,sn=0):
		matfilelist=[]

		for (dirpath,dirnames,filenames) in os.walk(self.matfolder):
			matfilelist.extend(filenames)
			break

		for i in [sn]:#range(1):
			print(matfilelist[i])
			matfile=self.matfolder+matfilelist[i]

			stim,resp=self.load_raw(matfile)
			sync=self.stim2sync(stim)	
			raster1=self.raw2raster(resp)

			rate=self.raster2rate(raster1)
			self.plot(matfile,stim,sync,resp,raster1,rate)

#########################################
############# test function #############
#########################################

	def plotPeakHist(self, log =1):
		matfilelist=[]

		for (dirpath,dirnames,filenames) in os.walk(self.matfolder):
			matfilelist.extend(filenames)
			break

		plt.figure(num=None, figsize=(8, 6), dpi=150, facecolor='w', edgecolor='k')
		
		hist_heatmap = numpy.zeros((0,500))

		for i in range(len(matfilelist)):#[sn]:#range(1):
			print(matfilelist[i])
			matfile=self.matfolder+matfilelist[i]
			
			stim,resp=self.load_raw(matfile)
			
			# peak_index = signal.find_peaks_cwt(resp,numpy.arange(1,3))

			peak_index = numpy.zeros((self.datalen),dtype=numpy.uint)
			for j in range(1,self.datalen-1):
				if (resp[j-1]<resp[j]>resp[j+1]) or (resp[j-1]>resp[j]<resp[j+1]) :
					peak_index[j] = 1

			# plt.plot(self.t,resp,'b')
			# plt.plot(self.t[peak_index==1],resp[peak_index==1],'r.')
			# n, x, _ = plt.hist(resp[peak_index==1],bins=50,histtype=u'step')
			# plt.plot(x+i)
			# print(n,x)
			hist, bin_edges = numpy.histogram(resp[peak_index==1], bins=numpy.linspace(0,5,num=500+1))
			# print(numpy.shape(hist))

			hist_heatmap=numpy.vstack((hist_heatmap,numpy.log(hist)))
			# plt.step(bin_edges[0:len(bin_edges)-1], numpy.log(hist))
			# print(bin_edges,hist)


		# plt.yscale('log', nonposy='clip')
		print(numpy.shape(hist_heatmap))
		plt.imshow(hist_heatmap, cmap='jet', interpolation='nearest')
		plt.xticks( numpy.linspace(0,500,num=10+1), numpy.linspace(0,5,num=10+1))
		plt.minorticks_on()
		plt.xlabel('Histogram of peak voltage (V)')
		plt.ylabel('Rec seq num')
		plt.title(self.matfolder.split('\\')[-2])

		if log==1 :
			if not os.path.exists(self.matfolder+"result\\"):
				os.makedirs(self.matfolder+"result\\")
			plt.savefig(self.matfolder + 'result\\hist.png')

		plt.show()

#########################################
############# main function #############
#########################################

if __name__ == "__main__":

	if len(sys.argv) > 1:
		if len(sys.argv) > 3:
			exp = JH_swing_rig_data_process(sys.argv[1],threshold=float(sys.argv[3]))
			print('Loading data at: '+sys.argv[1]+' with threshold at: '+ sys.argv[3]+' V')
		else:
			exp = JH_swing_rig_data_process(sys.argv[1],threshold=3.2)
			print('Loading data at: '+sys.argv[1])
	else:
		print('Error: Need 1st argument: data folder name ')
		exit()

	if len(sys.argv) > 2 :
		# print (len(sys.argv))
		if sys.argv[2] == 'hist' :
			exp.plotPeakHist()
		elif sys.argv[2] == 'snr' :
			exp.plotSNR(0)
		elif sys.argv[2] == 'get' :
			exp.getAllRaster_threshold()
			exp.getAllXYUV()		
		elif sys.argv[2] == 'raster' :
			exp.plotRaster()			
		elif sys.argv[2] == 'map' :
			exp.plotReceptiveField(quiver_scale=10)
		else:
			print('Error: undefined 2nd argument, try: hist, snr, raster, map')
	else:
		print('Error: Need 2nd argument, try: hist, snr, raster, map')
		exit()



	# exp.getAllXYUV2()	
	# exp.plotReceptiveField2()

	# exp.getAllRaster_dvdt()
