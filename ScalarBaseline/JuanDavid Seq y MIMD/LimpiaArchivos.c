//Analiza los archivos con texto alfabeto cualquiera, y pasa todos los caracteres a minuscula.


#include <stdio.h>
#include <ctype.h>
#include<stdlib.h>
#include<time.h>
#include<inttypes.h>
#include<string.h>
int main(void)
{	
	uint32_t size;
	//size = (uint32_t) atol(argv[1]);
	uint32_t i = 0;
	uint8_t num;
	FILE *archivo;
	char caracter;

	archivo = fopen("english","r");

	if (archivo == NULL)
        {
            printf("\nError de apertura del archivo. \n\n");
        }
        else
        {
            //printf("\nEl contenido del archivo de prueba es \n\n");
            while(((caracter = fgetc(archivo)) != EOF))
	    {
		//printf("%c",caracter);
		if(caracter != '\n'){
			i++;
		}
	    }
        }
        fclose(archivo);
	//printf("i vale %d\n", i);
	size = i;
///////////////////////////////////////////////////////////////////////////////////////////
	uint8_t *cadena = (uint8_t *)calloc(size+1, sizeof(uint8_t));
	i=0;
	archivo = fopen("english","r");
	
	if (archivo == NULL)
        {
            printf("\nError de apertura del archivo. \n\n");
        }
        else
        {
            //printf("\nEl contenido del archivo de prueba es \n\n");
            while(((caracter = fgetc(archivo)) != EOF))
	    {
		//printf("%c",caracter);
		if(caracter != '\n'){
			num = (uint8_t) caracter;
			cadena[i] = num;
			i++;
		}
	    }
        }
        fclose(archivo);
	//printf("i vale %d\n", i);
	cadena[size] = '\0';
 
	// Convertir cada char a mayúscula
	// usando toupper
	for (int indice = 0; cadena[indice] != '\0'; ++indice){
		cadena[indice] = tolower(cadena[indice]);
	}
	//printf("Cadena despues de ser convertida: %s\n", cadena);
 //////////////////////////////////////////////////////////////////////////////////////////
	i=0;
	char out0_file[]="english_";
	FILE *outfp0 = fopen(out0_file, "w");
  	if (outfp0 == NULL)
    	{
      		fprintf(stderr, "Error opening in file\n");
      		exit(1);
    	}
	//for(i=0;i<size;i++){
		//fprintf(outfp,"\t \tsufijo # %i pos: %i \n",i, Indexes[i]);
		fprintf(outfp0,"%s\n", &cadena[i]);		
	//}
        fclose(archivo);
	return 0;
}
