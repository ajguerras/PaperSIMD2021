#include<stdio.h>
#include<stdlib.h>
#include<time.h>
#include<inttypes.h>
#include<string.h>
#define BASE_BITS 8
#define BASE (1 << BASE_BITS)
#define MASK (BASE-1)
#define DIGITS(v, shift) (((v) >> shift) & MASK)
#define ENDCHAR 0

//#define archivos
//#define DEBUG
void Seq_lsd_radix_sort_(uint32_t size, uint8_t *cadena, uint32_t *Indexes, int64_t inf, int64_t sup);
void gen_data(uint8_t *cadena, uint32_t size);
void gen_Indexes(uint32_t *cadena, uint32_t size);
void SelectValidCharsMsc (uint8_t *InputText, uint32_t size, uint32_t Iter, uint32_t *Indexes, uint8_t *OutInt, uint8_t *HU, uint32_t *ID, uint32_t inf , uint32_t sup);
//void insertionSort(uint8_t *y, uint32_t *Indexes, int64_t inf, int64_t sup);
void procesar(uint32_t size, uint8_t *cadena, uint32_t *Indexes);
uint32_t GroupBuilder (uint8_t *InputText, uint32_t size, uint32_t Iter, uint32_t *Indexes, uint8_t *OutInt, int64_t *frontera, uint8_t *HU, uint32_t *ID);
int main(int argc, char* argv[]){
	
	struct timeval t1, t2;
  	double elapsedTime;	

	//int succP_= posix_memalign ((void **) &InputArr, 64, (sizePadded)*sizeof(unsigned char));

	uint32_t size;
	size = (uint32_t) atol(argv[1]);
	//size = 150000;
	printf("size= %ld\n",size);
	fflush(stdout);
	uint8_t *cadena = (uint8_t *)calloc(size+1, sizeof(uint8_t));
	if (cadena  == NULL) {
	/* Error al intentar reservar memoria */
		printf("error cadena\n");
	}
	uint32_t *Indexes = (uint32_t*) calloc(size+1, sizeof(uint32_t));//uint32
	if (Indexes  == NULL) {
	/* Error al intentar reservar memoria */
		printf("error Indexes\n");
	}
	
	gen_data(cadena, size);
	gen_Indexes(Indexes, size);
	#ifdef archivosIn
	char out0_file[]="sin_orden.dat";
	FILE *outfp0 = fopen(out0_file, "w");
  	if (outfp0 == NULL)
    	{
      		fprintf(stderr, "Error opening in file\n");
      		exit(1);
    	}
	for(uint32_t i=0;i<size;i++){
			fprintf(outfp0,"%s\n", &cadena[Indexes[i]]);		
	}
	fclose(outfp0);
	#endif
	gettimeofday(&t1, NULL);
	//printf("inicia\n"); fflush(stdout);
	procesar(size, cadena, Indexes);
	gettimeofday(&t2, NULL);
  	elapsedTime = (t2.tv_sec - t1.tv_sec) * 1000.0;      // sec to ms
  	elapsedTime += (t2.tv_usec - t1.tv_usec) / 1000.0;   // us to ms}
  	printf("Processing time: %5.3f ms\n", elapsedTime);
	

	/*for(i=0;i<size;i++){
		printf("%d")
	}*/
	#ifdef archivos
	uint32_t i;
	char out_file[20];
	sprintf(out_file, "Out_SACSeq512_%d.txt", size);
	FILE *outfp = fopen(out_file, "w");
  	if (outfp == NULL)
    	{
      		fprintf(stderr, "Error opening in file\n");
      		exit(1);
    	}
	for(i=0;i<size;i++){
			//fprintf(outfp,"\t \tsufijo # %i pos: %i \n",i, Indexes[i]);
			fprintf(outfp,"%s\n", &cadena[Indexes[i]]);		
	}
	fclose(outfp);
	#endif
	if(cadena){
		free(cadena);
	}
	printf("Cadena\n"); fflush(stdout);
	if(Indexes){
		free(Indexes);
	}
	printf("Indexes\n"); fflush(stdout);
	return 0;
}

void gen_data(uint8_t *cadena, uint32_t size){
	uint32_t i = 0;
	uint8_t num;
	FILE *archivo;
	char caracter;
	archivo = fopen("dna","r");
	
	if (archivo == NULL)
        {
            printf("\nError de apertura del archivo. \n\n");
        }
        else
        {
            //printf("\nEl contenido del archivo de prueba es \n\n");
            while(((caracter = fgetc(archivo)) != EOF) && (i<size))
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
	//for(i=size-1;i<=size;i++){printf("%d\n", cadena[i]);}
	//printf("%s\n", array);		
}
void gen_Indexes(uint32_t *cadena, uint32_t size){
	uint32_t i;
	for (i=0;i<size;i++){
		cadena[i] = i;
		#ifdef DEBUG 
		printf("%d\n", i);
		#endif
	}
}

void procesar(uint32_t size, uint8_t *cadena, uint32_t *Indexes){
	/*struct timeval t1, t2;
  	double elapsedTime;*/
	struct timeval t3, t4;
	double elapsedTime;

	 

	uint8_t *OutInt = (uint8_t *)calloc(size+1, sizeof(uint8_t));
	if (OutInt  == NULL) {
	/* Error al intentar reservar memoria */
		printf("error OutInt\n");
	}
	int64_t *frontera = (int64_t *)calloc(size+2, sizeof(int64_t));//para que pueda tener el -1 y aguantar el tamaño uint32_t
	if (frontera  == NULL) {
	/* Error al intentar reservar memoria */
		printf("error frontera\n");
	}
	uint8_t *HU =(uint8_t*)calloc(size+1, sizeof(uint8_t));
	if (HU  == NULL) {
	/* Error al intentar reservar memoria */
		printf("error HU\n");
	}
	uint32_t *ID =(uint32_t*)calloc(size+1, sizeof(uint32_t));/*arreglo para identificar los subgrupos que van apareciendo*/
	if (ID  == NULL) {
	/* Error al intentar reservar memoria */
		printf("error ID\n");
	}
	frontera[0] = -1;



	uint32_t x;
	uint32_t subgrupos=1;
	uint32_t Iter=0;
	uint8_t band=1;
	/*subgrupos = SelectValidCharsMsc(cadena, size, Iter, Indexes, OutInt, HU, ID, 0, size-1);				
	Seq_lsd_radix_sort_(size, OutInt, Indexes, frontera[0]+1, frontera[0+1]);
	Iter++;
	GroupBuilder (cadena, size,Iter, Indexes, OutInt, frontera, HU, ID);
	frontera [subgrupos] = size-1;*/
	do{
		//subgrupos = 1;
		//frontera[subgrupos] = size-1;
	//	SelectValidCharsMsc(cadena, size, Iter, Indexes, OutInt, HU, ID, 0, size-1);/*ultimo cambio*/
		//subgrupos = GroupBuilder (cadena, size, Iter, Indexes, OutInt, frontera, HU, ID);
		frontera [subgrupos] = size-1;
		band=1;
		gettimeofday(&t3, NULL);
		for(x=0;x<subgrupos;x++){
			if ( (frontera[x+1]-(frontera[x]+1)>0) ){	
				//subgrupos = SelectValidCharsMsc(cadena, size, Iter, Indexes, OutInt, HU, ID, 0, size-1);	
			//	gettimeofday(&t1, NULL);	
				SelectValidCharsMsc(cadena, size, Iter, Indexes, OutInt, HU, ID, frontera[x]+1, frontera[x+1]);		
				Seq_lsd_radix_sort_(size, OutInt, Indexes, frontera[x]+1, frontera[x+1]);
			/*	gettimeofday(&t2, NULL);
  				elapsedTime = (t2.tv_sec - t1.tv_sec) * 1000.0;      // sec to ms
  				elapsedTime += (t2.tv_usec - t1.tv_usec) / 1000.0;   // us to ms}
  				if((frontera[x+1]-(frontera[x]+1)) >2500){
  					printf("Processing time Select: %5.3f ms.", elapsedTime);
					printf("Tamaño Grupo: %li\n", frontera[x+1]-(frontera[x]+1));
				}*/
				
			}else {
				//uint32_t frontera[x+1];
				HU[Indexes[frontera[x+1]]]=1;
				//HU[Indexes[frontera[x]+1]]=1; //frontera[x] ????
			}
			//Es necesario llamar al constructor de grupos 
			band = band && HU[Indexes[frontera[x+1]]];
		}
		Iter++;
		subgrupos = GroupBuilder (cadena, size, Iter, Indexes, OutInt, frontera, HU, ID);
		gettimeofday(&t4, NULL);
  		elapsedTime = (t4.tv_sec - t3.tv_sec) * 1000.0;      // sec to ms
  		elapsedTime += (t4.tv_usec - t3.tv_usec) / 1000.0;   // us to ms}
  		//printf("Processing time: %5.3f ms, Iter=%d\n", elapsedTime, Iter-1);
	}while (!band);
	//printf("%d\n", Iter);

	if(frontera)
		free(frontera);
	printf("Frontera\n"); fflush(stdout);
	if(OutInt){
		free(OutInt);
	}
	printf("Out\n"); fflush(stdout);
	if(HU){
		free(HU);
	}
	printf("Hu\n"); fflush(stdout);
	if(ID){
		free(ID);
	}
	printf("Id\n"); fflush(stdout);


}



void Seq_lsd_radix_sort_(uint32_t size, uint8_t *cadena, uint32_t *Indexes, int64_t inf, int64_t sup){
//Este codigo debe limpiarse una vez que se decida si se ejecutara en paralelo o puramente secuencial

   // uint32_t *buffer = (uint32_t *) malloc(size*sizeof(uint32_t));
    uint32_t  *outIndexes = (uint32_t  *) malloc((sup-inf+1)*sizeof(uint32_t ));
   // if (buffer  == NULL) {
	/* Error al intentar reservar memoria */
	//	printf("error buffer\n");
   // }
    int total_digits = 8;//sizeof(uint32_t)*8; //DUDsizeA      int total_digits = sizeof(unsigned)*8;
    int64_t i;
    int32_t shift, cur_t;
    for(shift = 0; shift < total_digits; shift+=BASE_BITS) { //Teoricamentedebe descartarse. PROBAR HIP*
//generar versio limpia quitando sentencias OPENMP, comentarles
        uint32_t bucket[BASE] = {0};
        uint32_t local_bucket[BASE] = {0}; // size needed in each bucket
        //1st pass, scan whole and check the count
             for(i = inf; i <= sup; i++){
                local_bucket[DIGITS(cadena[i], shift)]++;
	//	printf("%d\n", DIGITS(data[i], shift));
             }
             for(i = 0; i < BASE; i++) {
                bucket[i] += local_bucket[i];
             }
             for (i = 1; i < BASE; i++) {
                bucket[i] += bucket[i - 1];
             }
            uint32_t nthreads = 1 ;
            uint32_t tid = 0;  
            for(cur_t = nthreads - 1; cur_t >= 0; cur_t--) {
                if(cur_t == tid) {
                    for(i = 0; i < BASE; i++) {
                        bucket[i] -= local_bucket[i];
                        local_bucket[i] = bucket[i];
                    }
                } //EndIf
            }//End for cur_t

            for(i = inf; i <= sup; i++) { //Final partial sorting
             //   buffer[local_bucket[DIGITS(cadena[i], shift)]++] = Indexes[i];
	          outIndexes[local_bucket[DIGITS(cadena[i], shift)]++] = Indexes[i];
            }
        //now move data
        //unsigned* tmp = data;//Indexes
	//uint32_t* tmp = Indexes;
        ///*Indexes*/data = buffer;
//	uint32_t h=0;
//	for(i=inf;i<=sup;i++){
//		Indexes[i] = buffer[h]; //Indexes = buffer; //memcpy //estos dos ciclos for en la version paralela deben desaparecer//
		
//		h++;
//	}
	//printf("sizeof= %d\n", sizeof(buffer));
//	h=0;
//	for(i=inf;i<=sup;i++){
//		buffer[h] = tmp[i];//buffer = tmp;/*OJO ACÁ*/
//		memcpy(buffer, tmp+inf, (sup-inf)<<2);
//		h++;
//	}
    } //EndFor shift
    memcpy(Indexes+inf, outIndexes, (sup-inf+1)<<2);
    //printf("sizeof uint32 = %d, outIndexes = %d\n",sizeof(uint32_t), sizeof(outIndexes));
   /* if(buffer){
    	free(buffer);
    }*/
} //end_Seq


void SelectValidCharsMsc (uint8_t *InputText, uint32_t size, uint32_t Iter, uint32_t *Indexes, uint8_t *OutInt, uint8_t *HU, uint32_t *ID, uint32_t inf , uint32_t sup){
	   uint32_t It;
	   
	



	   for (It=inf;It<=sup;++It){  //Para cada uno d elos caracterres validos    

		uint32_t ValidChar = Indexes[It] + Iter;

      		if(!(ValidChar >= size)){       //PAdding 
	             OutInt [It] = (uint8_t) InputText [  ValidChar ] ;  
		}else{
			OutInt[It] = 0;		//reevaluar
		}
	   }
}

uint32_t GroupBuilder (uint8_t *InputText, uint32_t size, uint32_t Iter, uint32_t *Indexes, uint8_t *OutInt, int64_t *frontera, uint8_t *HU, uint32_t *ID){
	   uint32_t subgrupos = 1;											;
	   int64_t It;//uint32_t It;
	   uint8_t aux=0;  	   /*aux= caracter valido previo*/
	   uint32_t aux_ID = ID[Indexes[0]];//se utiliza para saber el ID del sufijo antes de cambiarlo
   	   uint32_t cont_ID=0;
	   //int64_t iteracion;
	   if((Indexes[0]  +  Iter -1) >=size){ //Verifico la finalizacion de la cadena.
		aux = 0;
	   }else{
		aux = InputText [  Indexes[0]  +  Iter -1 ]; //Padding
	   }	
	  // printf("size -1 = %d\n", size-1);	
	   for (It=0;It<=size-1;++It){  //Para cada uno de los caracterres validos 
		uint32_t ValidChar = Indexes[It] + Iter;   
		int64_t itn = It-1;
		//recuperar a validchar de donde toque

		if(ValidChar>size){ //Inicia conformacion de grupos a partir de la 1 iteracion, cuando Se llega al final del sufijo . 
			//int64_t itn = It-1; //printf("itn = %d\n", itn);fflush(stdout);
			frontera[subgrupos] = itn; //(int64_t) (It -1);   //Creamos la nueva frontera
			//	   printf("%d\n", subgrupos);fflush(stdout);
			//HU[It] = 1; //agregada
			if(It==0){  //El 1er Caracter valido es invalido entonces se crea una frontera
				frontera[subgrupos] = It;//(int64_t)(It);
			}
			subgrupos++;
			aux = 0;
			//printf("%d Termina ValidChar>Size\n", It);fflush(stdout);
		}else if((InputText [  ValidChar -1 ] != aux) || (HU[Indexes[It]]==1)){ /*||((carcteres ante son iguales)&&(Grupo[sufijo actual]!=Grupo[sufijo anterio]))*///(caracter diferente) (no es la 1st iteracion)(sufijo ya ha sido ordenado)
			//printf("Entra\n"); fflush(stdout);
			frontera[subgrupos] =itn;// (int64_t)It -1;
			if(It==0){
				frontera[subgrupos] = It;//(int64_t)It;
			}
			subgrupos++;
			//printf("primero\n"); 			fflush(stdout);
			aux = (uint8_t) InputText [  ValidChar -1 ];
			//printf("segundo\n");			fflush(stdout);
			aux_ID = ID[Indexes[It]];                 
			cont_ID++;
			ID[Indexes[It]] = cont_ID;/*aux_ID guarda el valor del subgrupo antes de actualizarlo, para compararlo con los sufijos siguientes */
			if(HU[Indexes[It]]==1){
				aux=(0);
			}
		}else if(((InputText [  ValidChar -1 ] == aux) && (Iter>0))){
			//printf("Entra2\n");			fflush(stdout);
			if(aux_ID==ID[Indexes[It]]){
				ID[Indexes[It]] = cont_ID;
			}else{
				frontera[subgrupos] = itn;//(int64_t)It -1;
				if(It==0){
					frontera[subgrupos] = It;//(int64_t)It;
				}
				subgrupos++;
				aux = (uint8_t) InputText [  ValidChar -1 ];
				aux_ID = ID[Indexes[It]];
				cont_ID++;
				ID[Indexes[It]] = cont_ID; 
				if(HU[Indexes[It]]==1){
					aux=(0);
				}	
			}
		}
   
 	   }
	//   printf("Termina toda la funcion\n");fflush(stdout);

	   return subgrupos;    
}
