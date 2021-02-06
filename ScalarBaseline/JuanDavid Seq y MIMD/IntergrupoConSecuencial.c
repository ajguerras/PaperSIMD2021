#include<stdio.h>
#include<stdlib.h>
#include<time.h>
#include<inttypes.h>
#include<string.h>
#include<omp.h>
#define BASE_BITS 8
#define BASE (1 << BASE_BITS)
#define MASK (BASE-1)
#define DIGITS(v, shift) (((v) >> shift) & MASK)
#define ENDCHAR 0
#define HIGHESTCHAR 0
#define TAM_GROUP 5000
#define SEQ_THRESHOLD 3500

#define MIN(X, Y) (((X) < (Y)) ? (X) : (Y))

//#define archivos
//#define DEBUG
//#define DEBUG_IS
void gen_data(uint8_t *cadena, uint32_t size);
void InitializeSuffixMIMD (uint32_t SizeInput, uint32_t *Suffixes, uint8_t NT, uint32_t ChunkSize);
void SelectValidCharMIMD (uint8_t *InputText, uint32_t Size, uint32_t currentIter, uint32_t *Indexes, uint8_t *OutInt, uint32_t Inf, uint32_t Sup, uint8_t Nthreads );
void omp_lsd_radix_sortInd(uint32_t n, uint8_t *data, uint32_t *Ind, uint32_t inf, uint32_t sup, uint8_t nthreads);
void SelectValidCharsMsc (uint8_t *InputText, uint32_t size, uint32_t Iter, uint32_t *Indexes, uint8_t *OutInt, uint8_t *HU, uint32_t inf , uint32_t sup);
void gen_Indexes(uint32_t *cadena, uint32_t size);
void Seq_lsd_radix_sort_(uint32_t size, uint8_t *cadena, uint32_t *Indexes, uint32_t inf, uint32_t sup);
void procesar(uint32_t size, uint8_t *cadena, uint32_t *Indexes, uint8_t nthreads);
uint32_t GroupBuilder (uint8_t *InputText, uint32_t size, uint32_t Iter, uint32_t *Indexes, uint8_t *OutInt, uint32_t *frontera, uint8_t *HU, uint32_t inf, uint32_t sup);
void seq(uint32_t size, uint8_t *cadena, uint32_t *Indexes, uint32_t superior, uint32_t inferior, uint32_t Iter, uint8_t nthreads, uint32_t *frontera, uint8_t *OutInt, uint8_t *HU);
int main(int argc, char* argv[]){
	
	struct timeval t1, t2;
  	double elapsedTime;	

	//int succP_= posix_memalign ((void **) &InputArr, 64, (sizePadded)*sizeof(unsigned char));
	/*FILE *archivo;
	char caracter;
	uint32_t tam = 0;
	archivo = fopen("dna","r");
	
	if (archivo == NULL)
        {
            printf("\nError de apertura del archivo. \n\n");
        }
        else
        {
            //printf("\nEl contenido del archivo de prueba es \n\n");
            while((caracter = fgetc(archivo)) != EOF)
	    {
		//printf("%c",caracter);
		if(caracter != '\n')
		tam++;
	    }
        }
        fclose(archivo);*/
	//printf("El tamaño es %d\n", tam);

	uint32_t size;
	uint8_t nthreads;
	size = (uint32_t) atol(argv[1]); 
	//size = tam;
	nthreads = (uint8_t) atoi(argv[2]);
	//size = 150000;
	printf("size= %ld threads=%ld\n",size, nthreads);
	//fflush(stdout);
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
	uint32_t ChunkSize=(uint32_t) (size/nthreads)+1;
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
	//printf("Iniciando a ordenar\n");
	gettimeofday(&t1, NULL);
	//printf("inicia\n"); fflush(stdout);
	//printf("%s\n", cadena);
	procesar(size, cadena, Indexes, nthreads);
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
	archivo = fopen("english_","r");
	
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

void SelectValidCharMIMD (uint8_t *InputText, uint32_t Size, uint32_t currentIter, uint32_t *Indexes, uint8_t *OutInt, uint32_t Inf, uint32_t Sup, uint8_t Nthreads ){

   uint32_t ChunkSize=(uint32_t) ((Sup-Inf)/Nthreads)+1;
   #pragma omp parallel  num_threads(Nthreads) shared(ChunkSize) //firstprivate(Res,Displacement)
   { 
		
	int tid = omp_get_thread_num(); //Pido Mi Id,
  	uint32_t Inicio= (tid*ChunkSize)+Inf;
	



	uint32_t ValidChar;
	uint32_t Fin= MIN(Sup, (Inicio + ChunkSize)); // OJO(Si llega a presentar error habilitar este:) if (tid==(Nthreads-1))  Fin=Sup ;
	//printf ("Soy %d, Inicio =%lu, Fin = %lu, It= %lu, Sup = %lu\n",tid,Inicio,Fin, It, Sup);fflush(stdout);
        for (uint32_t auxIt=Inicio;auxIt<=Fin;auxIt++){   
		ValidChar = Indexes[auxIt] + currentIter;
		if(!(ValidChar >= Size)){       //PAdding 
		    OutInt [auxIt] = InputText [  ValidChar ] ;  
		}else{
		    OutInt[auxIt] = HIGHESTCHAR;		//reevaluar, Caracter Invalido. Debe ser cero o debe ser un numero muy grande para que no afecte el prefix sum
		}
	}

	 //printf ("Soy %d SVC ITERATIVO OK\n",tid);fflush(stdout);
   }//end pragma
 
        
}//EndSelectValidChars

void InitializeSuffixMIMD (uint32_t SizeInput, uint32_t *Suffixes, uint8_t NT, uint32_t ChunkSize){  //SizeInput is measured from zero, so when calling passes SizeInput-1

   // __m512i Displacement,Res=MASK_AVX512_0to15; 		  //1. Fill  register 0 - 15
   
   #pragma omp parallel shared(ChunkSize) num_threads(NT) 
   {     
           int tid = omp_get_thread_num(); //Pido Mi Id,
           //Cada hilo debe calcular el desplazamiento,  y el resultado es My ID * Chunksize
	   uint32_t Inicio= tid*ChunkSize;
	   uint32_t Fin = MIN(SizeInput, (Inicio + ChunkSize));  // considerr el caso del ultimo chunk
	   for (uint32_t x=Inicio;x<Fin; x++  ){ 
			Suffixes[x]=x;

	   }//endfor
   }
    #ifdef DEBUG_IS
    	for (uint32_t i=0; i<SizeInput;++i)         { if (Suffixes[i]!=i) printf ("***ERROR SuffixPos[%d]= %d\n  ***********",i, Suffixes[i]) ; }
      
  	/*for (int i=0; i<sizeInArrF;++i) {
            if (SPos[i]!=(sizeInArr-i-1))   { 
		printf ("********Main Error Initializing SuffixPos[%d]= %d\n  ***********",i, SPos[i]) ; 
		SPos[i]=(sizeInArr-i); 
            }
         }
       int c = getchar ();*/
    #endif

}//End Initialize SuffixSIMD

void Seq_lsd_radix_sort_(uint32_t size, uint8_t *cadena, uint32_t *Indexes, uint32_t inf, uint32_t sup){
//Este codigo debe limpiarse una vez que se decida si se ejecutara en paralelo o puramente secuencial

   // uint32_t *buffer = (uint32_t *) malloc(size*sizeof(uint32_t));
    uint32_t  *outIndexes = (uint32_t  *) malloc((sup-inf+1)*sizeof(uint32_t ));
   // if (buffer  == NULL) {
	/* Error al intentar reservar memoria */
	//	printf("error buffer\n");
   // }
    int total_digits = 8;//sizeof(uint32_t)*8; //DUDsizeA      int total_digits = sizeof(unsigned)*8;
    uint32_t i;
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
            uint8_t nthreads = 1 ;
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

void SelectValidCharsMsc (uint8_t *InputText, uint32_t size, uint32_t Iter, uint32_t *Indexes, uint8_t *OutInt, uint8_t *HU, uint32_t inf , uint32_t sup){
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

	

void procesar(uint32_t size, uint8_t *cadena, uint32_t *Indexes, uint8_t nthreads){
	/*struct timeval t1, t2;
  	double elapsedTime;*/


	uint8_t *OutInt = (uint8_t *)calloc(size+1, sizeof(uint8_t));
	if (OutInt  == NULL) {
	/* Error al intentar reservar memoria */
		printf("error OutInt\n");
	}
	uint32_t *frontera = (uint32_t *)calloc(size+2, sizeof(uint32_t));//para que pueda tener el -1 y aguantar el tamaño uint32_t
	if (frontera  == NULL) {
	/* Error al intentar reservar memoria */
		printf("error frontera\n");
	}
	uint8_t *HU =(uint8_t*)calloc(size+1, sizeof(uint8_t));
	if (HU  == NULL) {
	/* Error al intentar reservar memoria */
		printf("error HU\n");
	}
	//uint32_t *ID =(uint32_t*)calloc(size+1, sizeof(uint32_t));/*arreglo para identificar los subgrupos que van apareciendo*/
	//if (ID  == NULL) {
	/* Error al intentar reservar memoria */
	//	printf("error ID\n");
	//}
	//frontera[0] = 1;


	//uint32_t tam_group=0;
	uint32_t x;
	uint32_t subgrupos=1;
	uint32_t Iter=0, inf, sup;
	uint8_t band=1; //bandera=1;
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
		frontera [size-1] = 1;
		band=1;
		inf=0;
		#pragma omp parallel num_threads(nthreads)
		{
			uint8_t id = omp_get_thread_num();
			if(id==0)
			{
				for (x=0; x<size; x++){//Lanzo intergrupo
					/*if(frontera[x] == 1 && bandera==0){
						inf=x;bandera=1;
					}
					else*/ if(frontera[x] == 1 /*&& bandera==1*/){
						//if(Iter==6) printf("i = %d\n", x);
						sup=x;//bandera=0; inf=sup+1;
						//if(sup<0 || sup>=size) 			printf("prueba sup=%d\n", sup);
						
						uint32_t my_inf; uint32_t my_sup; 
						my_inf = inf;my_sup = sup; inf=sup+1;
						if(my_sup-my_inf>0){
							#pragma omp task firstprivate (my_inf, my_sup)
							{
								//struct timeval t3, t4;
								//gettimeofday(&t3, NULL);
								//double elapsedTime; 
								//if(inf>=size){printf("my_inf= %d, my_sup= %d, Iter= %d\n", my_inf, my_sup, Iter);}
								#ifdef DEBUG3
								if((my_sup-my_inf) > (size/5))
								printf("Hola soy %d tam = %d, sup=%d, inf=%d\n",omp_get_thread_num(),my_sup-my_inf, my_sup, my_inf);	
								#endif
								//if ( (my_sup-my_inf)<TAM_GROUP) {tam_group ++;}		
								if ( (my_sup-my_inf)>SEQ_THRESHOLD ){	
									//if(Iter==6) 
									//	printf("Entra threshold\n");
										SelectValidCharsMsc(cadena, size, Iter, Indexes, OutInt, HU, my_inf, my_sup);	
									//if(Iter==6) 
									//	printf("sale threshold\n");	
									#ifdef DEBUG2
									printf("cadena antes %s\n",OutInt);	
									#endif
									#ifdef DEBUG2
									for(int i=0;i<size;i++){printf("Indexes omp=%d\n", Indexes[i]);} 	
									#endif
										Seq_lsd_radix_sort_(size, OutInt, Indexes, my_inf, my_sup);
								
									#ifdef DEBUG2
									printf("cadena despues %s\n",OutInt);	
									#endif
									#ifdef DEBUG2
									for(int i=0;i<size;i++){printf("Indexes omp=%d\n", Indexes[i]);} 	
									#endif
								}else if(((my_sup-my_inf)<= SEQ_THRESHOLD) && ((my_sup-my_inf)>0)){
										//printf("Entrando seq\n");
									//if(Iter==6) 
									//	printf("Entra");
									seq(size, cadena, Indexes, my_sup, my_inf, Iter, nthreads, frontera, OutInt, HU);
									//if(Iter==6) 
									//	printf("sale");
									//printf("saliendo seq\n");
								}
								else {
									//uint32_t frontera[x+1];
									HU[Indexes[x]]=1;
									//HU[Indexes[frontera[x]+1]]=1; //frontera[x] ????
								}
								#pragma omp critical   ////OJJJOOOOOO
								{
								band = band && HU[Indexes[x]];
								}
								//uint8_t my_Iter = inf;my_Iter++;
								subgrupos = GroupBuilder (cadena, size, Iter+1, Indexes, OutInt, frontera, HU, my_inf, my_sup);
								//if(Iter==6)
								//	printf("Saliendo GroupBuilder\n");
								#ifdef DEBUG2
								for(int i=0;i<size;i++){printf("Frontera[%d]=%d\n", i, frontera[i]);} 	
								#endif
								//gettimeofday(&t4, NULL);
	  							//elapsedTime = (t4.tv_sec - t3.tv_sec) * 1000.0;      // sec to ms
	  							//elapsedTime += (t4.tv_usec - t3.tv_usec) / 1000.0;   // us to ms}
							//	if((my_sup-my_inf) > (size/5))
	  						//	printf("Processing time: %5.3f ms, hilo=%d, Iter=%d\n", elapsedTime, omp_get_thread_num(), Iter);
							}
						}else{HU[Indexes[x]] = 1;}
						//if(Iter==6) printf("Sale del task\n");
					} //if frontera==1
					//if(Iter==6) printf("i = %d\n", x);
				}//Lanzo intergrupo
				//if(Iter==6) printf("Sale del for\n");
			}
			#pragma omp barrier
			#ifdef DEBUG2
			printf("Sincronización de todos los hilos\n");
			#endif
		} 
		//if(Iter==6) 
		//	printf("Sincronizando hilo\n");
		//printf("Iter = %d\n", Iter);
		Iter++;
	}while (!band);//(Iter<2);
	//printf("#grupos menores a %d = %d\n", TAM_GROUP, tam_group);
	printf("Inter+Seq\n");
	if(frontera){
		free(frontera);
		printf("Frontera\n"); fflush(stdout);
	}
	if(OutInt){
		free(OutInt);
		printf("Out\n"); fflush(stdout);
	}
	if(HU){
		free(HU);
		printf("Hu\n"); fflush(stdout);
	}
	//if(ID){
	//	free(ID);
	//	printf("Id\n"); fflush(stdout);
	//}
}

uint32_t GroupBuilder (uint8_t *InputText, uint32_t size, uint32_t Iter, uint32_t *Indexes, uint8_t *OutInt, uint32_t *frontera, uint8_t *HU, uint32_t inf, uint32_t sup){
	   uint32_t subgrupos = 1;											
	   uint32_t It;//uint32_t It;
	   uint8_t aux=0;  	   /*aux= caracter valido previo*/
	//   uint32_t aux_ID = ID[Indexes[inf]];//se utiliza para saber el ID del sufijo antes de cambiarlo
   	   //uint32_t cont_ID=0;
	   //int64_t iteracion;
	   if((Indexes[inf]  +  Iter -1) >=size){ //Verifico la finalizacion de la cadena.
		aux = 0;
	   }else{
		aux = InputText [  Indexes[inf]  +  Iter -1 ]; //Padding
		//printf("Aux=%d\n", aux);
	   }	
	  // printf("size -1 = %d\n", size-1);	
	   for (It=inf;It<=sup;++It){  //Para cada uno de los caracterres validos 
		uint32_t ValidChar = Indexes[It] + Iter;  
		#ifdef DEBUG2
		printf("ValidCharGroupBuilder = %d\n", ValidChar); 
		printf("aux = %c ITer=%d\n", aux, Iter); 
		#endif
		uint32_t itn = It-1;
		//recuperar a validchar de donde toque

		if(ValidChar>size){ //Inicia conformacion de grupos a partir de la 1 iteracion, cuando Se llega al final del sufijo . 
			//int64_t itn = It-1; //printf("itn = %d\n", itn);fflush(stdout);
			if(It==inf){  //El 1er Caracter valido es invalido entonces se crea una frontera
				frontera[It] = 1;//(int64_t)(It);
				#ifdef DEBUG2
				printf(">size frontera[%d] = 1\n", It); 
				#endif
			}else{
				frontera[itn] = 1; //(int64_t) (It -1);   //Creamos la nueva frontera
				#ifdef DEBUG2
				printf(">size frontera[%d] = 1\n", itn); 
				#endif
			}
			subgrupos++;
			aux = 0;
			//printf("%d Termina ValidChar>Size\n", It);fflush(stdout);
		}else if((InputText [  ValidChar -1 ] != aux) || (HU[Indexes[It]]==1)){ /*||((carcteres ante son iguales)&&(Grupo[sufijo actual]!=Grupo[sufijo anterio]))*///(caracter diferente) (no es la 1st iteracion)(sufijo ya ha sido ordenado)
			//printf("Entra\n"); fflush(stdout);
			if(It==inf){
				frontera[It] = 1;//(int64_t)It;
				#ifdef DEBUG2
				printf("letra!= o HU ,frontera[%d] = 1\n", It); 
				#endif 
			}else{
				frontera[itn] =1;// (int64_t)It -1;
				#ifdef DEBUG2
				printf("letra!= o HU ,frontera[%d] = 1\n", itn); 
				#endif
			}
			subgrupos++;

			aux = (uint8_t) InputText [  ValidChar -1 ];

		//l	aux_ID = ID[Indexes[It]];                 
		//l	cont_ID++;
		//l	ID[Indexes[It]] = cont_ID;/*aux_ID guarda el valor del subgrupo antes de actualizarlo, para compararlo con los sufijos siguientes */
			if(HU[Indexes[It]]==1){
				aux=(0);
			}
		}/*l else if(((InputText [  ValidChar -1 ] == aux) && (Iter>0))){
			//printf("Entra2\n");			fflush(stdout);
			if(aux_ID==ID[Indexes[It]]){
				ID[Indexes[It]] = cont_ID;
			}else{
				frontera[itn] = 1;//(int64_t)It -1;
				#ifdef DEBUG2
				printf("ID frontera[%d] = 1\n", itn); 
				#endif
				if(It==inf){
					frontera[It] = 1;//(int64_t)It;
					#ifdef DEBUG2
					printf("ID frontera[%d] = 1\n", It); 
					#endif
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
		}  l*/
   
 	   }
	//   printf("Termina toda la funcion\n");fflush(stdout);

	   return subgrupos;    
}


void seq(uint32_t size, uint8_t *cadena, uint32_t *Indexes, uint32_t superior, uint32_t inferior, uint32_t Iter, uint8_t nthreads, uint32_t *frontera, uint8_t *OutInt, uint8_t *HU){
	

	uint32_t tam_group=0;
	uint32_t x;
	uint32_t subgrupos=1;
	uint32_t inf, sup;
	uint8_t band=1; //bandera=1;
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
		frontera [superior] = 1;
		band=1;
		inf=inferior;
	//	#pragma omp parallel num_threads(nthreads)
	//	{
	//		uint8_t id = omp_get_thread_num();
	//		if(id==0)
	//		{
				for (x=inferior; x<=superior; x++){//Lanzo intergrupo
					/*if(frontera[x] == 1 && bandera==0){
						inf=x;bandera=1;
					}
					else*/ if(frontera[x] == 1 /*&& bandera==1*/){
						sup=x;//bandera=0; inf=sup+1;
						//if(sup<0 || sup>=size) 			printf("prueba sup=%d\n", sup);
						
						uint32_t my_inf; uint32_t my_sup; 
						my_inf = inf;my_sup = sup; inf=sup+1;
						//#pragma omp task firstprivate (my_inf, my_sup)
						//{
							//struct timeval t3, t4;
							//gettimeofday(&t3, NULL);
							//double elapsedTime; 
							//if(inf>=size){printf("my_inf= %d, my_sup= %d, Iter= %d\n", my_inf, my_sup, Iter);}
							#ifdef DEBUG3
							if((my_sup-my_inf) > (size/5))
							//printf("Hola soy %d tam = %d, sup=%d, inf=%d\n",omp_get_thread_num(),my_sup-my_inf, my_sup, my_inf);	
							#endif
							if ( (my_sup-my_inf)<TAM_GROUP) {tam_group ++;}		
							if ( (my_sup-my_inf)>0 ){
									SelectValidCharsMsc(cadena, size, Iter, Indexes, OutInt, HU, my_inf, my_sup);			
								#ifdef DEBUG2
								printf("cadena antes %s\n",OutInt);	
								#endif
								#ifdef DEBUG2
								for(int i=0;i<size;i++){printf("Indexes omp=%d\n", Indexes[i]);} 	
								#endif
									Seq_lsd_radix_sort_(size, OutInt, Indexes, my_inf, my_sup);
								
								#ifdef DEBUG2
								printf("cadena despues %s\n",OutInt);	
								#endif
								#ifdef DEBUG2
								for(int i=0;i<size;i++){printf("Indexes omp=%d\n", Indexes[i]);} 	
								#endif
							}else {
								//uint32_t frontera[x+1];
								HU[Indexes[x]]=1;
								//HU[Indexes[frontera[x]+1]]=1; //frontera[x] ????
							}
							//#pragma omp critical   ////OJJJOOOOOO
							//{
							band = band && HU[Indexes[x]];
							//}
							//uint8_t my_Iter = inf;my_Iter++;
							subgrupos = GroupBuilder (cadena, size, Iter+1, Indexes, OutInt, frontera, HU, my_inf, my_sup);
							#ifdef DEBUG2
							for(int i=0;i<size;i++){printf("Frontera[%d]=%d\n", i, frontera[i]);} 	
							#endif
							//gettimeofday(&t4, NULL);
  							//elapsedTime = (t4.tv_sec - t3.tv_sec) * 1000.0;      // sec to ms
  							//elapsedTime += (t4.tv_usec - t3.tv_usec) / 1000.0;   // us to ms}
							//if((my_sup-my_inf) > (size/5))
  							//printf("Processing time: %5.3f ms, hilo=%d, Iter=%d\n", elapsedTime, omp_get_thread_num(), Iter);
						//}
					}
				}//Lanzo intergrupo

			//}
			//#pragma omp barrier
			#ifdef DEBUG2
			printf("Sincronización de todos los hilos\n");
			#endif
		//}
		
		Iter++;
	}while (!band); //(Iter<2);
	//printf("#grupos menores a %d = %d\n", TAM_GROUP, tam_group);
}
