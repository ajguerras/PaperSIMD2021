import os
#sim='pruebas_finales.c'
#ex='emb'
#varying number of Threads
threads=[2,4,8,16,24,32,48,64]
size=[1000000,5000000]
#cmd='gcc -o '+ex+' '+sim+' -fopenmp'
#print cmd
#os.system(cmd)

#for siz in size:
#	for thread in threads:
#		cmd='./'+ex+' '+str(siz)+' '+str(thread) +' >> salida.txt'
#		print cmd
#		os.system(cmd)

#sim='SecuencialVersion2.c'
#ex='seq'
#cmd='gcc -o '+ex+' '+sim+' -fopenmp'
#print cmd
#os.system(cmd)

#for siz in size:
#	cmd='./'+ex+' '+str(siz)+' 1 >> salida.txt'
#	print cmd
#	os.system(cmd)

#sim='intergrupo.c'
#ex='inter'
#cmd='gcc -o '+ex+' '+sim+' -fopenmp'
#print cmd
#os.system(cmd)

#for siz in size:
#	for thread in threads:
#		cmd='./'+ex+' '+str(siz)+' '+str(thread) +' >> salida.txt'
#		print cmd
#		os.system(cmd)

#sim='intragrupo.c'
#ex='intra'
#cmd='gcc -o '+ex+' '+sim+' -fopenmp'
#print cmd
#os.system(cmd)

#for siz in size:
#	for thread in threads:
#		cmd='./'+ex+' '+str(siz)+' '+str(thread) +' >> salida.txt'
#		print cmd
#		os.system(cmd)

sim='intra_seq.c'
ex='intra_seq'
cmd='gcc -o '+ex+' '+sim+' -fopenmp'
print cmd
os.system(cmd)

for siz in size:
	for thread in threads:
		cmd='./'+ex+' '+str(siz)+' '+str(thread) +' >> salida.txt'
		print cmd
		os.system(cmd)


