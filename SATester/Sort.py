#Automaticamente verifica todos los txt en el directorio. sin dependencias en otro archivo externo
#python Sort.py
#Esta version carga todos los sufijos, los ordena  y compara ambas listas.
#Existe otra version que se construyo con camilo usando compando python para comparar contenido de dos archivos.
#Deberia existir una tercera version que compare solamente los indices de los sufijos entre dos archivos. 

import os
lista=[]
lista_ordenada=[]
#ile = open('sin_orden.dat', 'r')
#for linea in file:
    #print(linea)
 #   lista.append(linea)
#file.close()
#lista.sort()
ruta = os.getcwd()
lstDir = os.walk(ruta)

for root, dirs, files in lstDir:
	for fichero in files:
		(nombreFichero, extension) = os.path.splitext(fichero)

		if(extension == ".txt"):
			a,b,c = nombreFichero.split("_")
			if(a == "Out"):
				print(fichero)
				file2 = open(fichero, 'r')
				for linea2 in file2:
					lista_ordenada.append(linea2)
					lista.append(linea2)
				file2.close()
				lista_ordenada.sort()
				print(lista==lista_ordenada)
			del lista[:]
			del lista_ordenada[:]
			#lista.clear()
			#lista_ordenada.clear()
			#print(nombreFichero)
#file2 = open('orden.dat', 'r')
#for linea2 in file2:
#    lista_ordenada.append(linea2)
#    lista.append(linea2)
#file2.close()
#lista_ordenada.sort()

#print(lista==lista_ordenada)
