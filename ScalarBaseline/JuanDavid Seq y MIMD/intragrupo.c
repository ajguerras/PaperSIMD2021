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

#define MIN(X, Y) (((X) < (Y)) ? (X) : (Y))

//#define archivos
//#define DEBUG
//#define DEBUG_IS
void gen_data(uint8_t *cadena, uint32_t size);
void InitializeSuffixMIMD (uint32_t SizeInput, uint32_t *Suffixes, uint8_t NT, uint32_t ChunkSize);
void SelectValidCharMIMD (uint8_t *InputText, uint32_t Size, uint32_t currentIter, uint32_t *Indexes, uint8_t *OutInt, int64_t Inf, int64_t Sup, uint8_t Nthreads );
void omp_lsd_radix_sortInd(uint32_t n, uint8_t *data, uint32_t *Ind, int64_t inf, int64_t sup, uint8_t nthreads);
void procesar(uint32_t size, uint8_t *cadena, uint32_t *Indexes, uint8_t nthreads);
uint32_t GroupBuilder (uint8_t *InputText, uint32_t size, uint32_t Iter, uint32_t *Indexes, uint8_t *OutInt, int64_t *frontera, uint8_t *HU, uint32_t *ID);
int main(int argc, char* argv[]){
	
	struct timeval t1, t2;
  	double elapsedTime;	

	//int succP_= posix_memalign ((void **) &InputArr, 64, (sizePadded)*sizeof(unsigned char));

	uint32_t size;
	uint8_t nthreads;
	size = (uint32_t) atol(argv[1]);
	nthreads = (uint32_t) atoi(argv[2]);
	//size = 150000;
	//printf("size= %ld\n",size);
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
	uint32_t ChunkSize=(uint32_t) (size/nthreads)+1;
	InitializeSuffixMIMD(size, Indexes, nthreads, ChunkSize);
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



void SelectValidCharMIMD (uint8_t *InputText, uint32_t Size, uint32_t currentIter, uint32_t *Indexes, uint8_t *OutInt, int64_t Inf, int64_t Sup, uint8_t Nthreads ){

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
    	for (int i=0; i<SizeInput;++i)         { if (Suffixes[i]!=i) printf ("***ERROR SuffixPos[%d]= %d\n  ***********",i, Suffixes[i]) ; }
      
  	/*for (int i=0; i<sizeInArrF;++i) {
            if (SPos[i]!=(sizeInArr-i-1))   { 
		printf ("********Main Error Initializing SuffixPos[%d]= %d\n  ***********",i, SPos[i]) ; 
		SPos[i]=(sizeInArr-i); 
            }
         }
       int c = getchar ();*/
    #endif

}//End Initialize SuffixSIMD


void procesar(uint32_t size, uint8_t *cadena, uint32_t *Indexes, uint8_t nthreads){

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

	//struct timeval t1, t2;
  	//double elapsedTime;
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
		/*SelectValidCharMIMD(cadena, size, Iter, Indexes, OutInt, 0, size-1, nthreads);
		printf("%s\n", OutInt);
		omp_lsd_radix_sortInd(size, OutInt, Indexes, 0, size);*/
	//	for(int i=0;i<size;i++){/*printf("%d\n", Indexes[i]);*/	printf("Data pos Segun Ind[%u]=%u,  %u\n ",i,Indexes[i],  cadena[Indexes[i]]);}		
	//	void inline SelectValidCharMIMD (uint8_t *InputText, uint32_t Size, uint32_t currentIter, uint32_t *Indexes, uint8_t *OutInt, int64_t Inf, int64_t Sup, int Nthreads );
		//subgrupos = GroupBuilder (cadena, size, Iter, Indexes, OutInt, frontera, HU, ID);
		frontera [subgrupos] = size-1;
		band=1;
	//	#pragma omp parallel for num_threads(nthreads) shared(frontera, cadena, size, band, Indexes)
		//gettimeofday(&t1, NULL);
		for(x=0;x<subgrupos;x++){
			
			if ( (frontera[x+1]-(frontera[x]+1)>0) ){
			/*	uint32_t tam = (frontera[x+1]-(frontera[x]+1)+1);
				printf("tam= %d, x=%d\n", tam, x);
				uint8_t *OutIntp = (uint8_t *)calloc(tam+1, sizeof(uint8_t));
				if (OutIntp  == NULL) {
					printf("error OutInt\n");
				}*/
				//subgrupos = SelectValidCharsMsc(cadena, size, Iter, Indexes, OutInt, HU, ID, 0, size-1);
					
				SelectValidCharMIMD(cadena, size, Iter, Indexes, OutInt, frontera[x]+1, frontera[x+1], nthreads);
			//	printf("%s\n", OutInt);			
				omp_lsd_radix_sortInd(size, OutInt, Indexes, frontera[x]+1, frontera[x+1]+1, nthreads);
	//			for(int i=0;i<size;i++){printf("%d\n", Indexes[i]);/*	printf("Data pos Segun Ind[%u]=%u,  %u\n ",i,Indexes[i],  cadena[Indexes[i]]);*/}		
//				for(int i=0;i<size;i++){printf("%d\n", Indexes[i]);}
				
			/*	if((frontera[x+1]-(frontera[x]+1)) >3500){
  					printf("Processing time Select, omp: %5.3f ms.", elapsedTime);
					printf("Tamaño Grupo: %li\n", frontera[x+1]-(frontera[x]+1));
				}*/
								
			}else {
				//uint32_t frontera[x+1];
				HU[Indexes[frontera[x+1]]]=1;
				//HU[Indexes[frontera[x]+1]]=1; //frontera[x] ????
			}
			//Es necesario llamar al constructor de grupos 
	//		#pragma omp critical
			band = band && HU[Indexes[frontera[x+1]]];
		}
		
		Iter++;
		subgrupos = GroupBuilder (cadena, size, Iter, Indexes, OutInt, frontera, HU, ID);
		//gettimeofday(&t2, NULL);
  		//elapsedTime = (t2.tv_sec - t1.tv_sec) * 1000.0;      // sec to ms
  		//elapsedTime += (t2.tv_usec - t1.tv_usec) / 1000.0;   // us to ms}
		//printf("Processing time: %5.3f ms, Iter=%d\n", elapsedTime, Iter-1);
	//	printf("subgrupos= %d\n", subgrupos);
	}while (!band);
	//printf("Iter = %d\n", Iter);
	printf("Intragrupo\n");
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

void omp_lsd_radix_sortInd(uint32_t n, uint8_t *data, uint32_t *Ind, int64_t inf, int64_t sup, uint8_t nthreads ) { //Cerrado , abierto
    //unsigned * buffer = (unsigned *) malloc(n*sizeof(unsigned));
    uint32_t  *outIndexes = (uint32_t  *) malloc((sup-inf)*sizeof(uint32_t ));
    int total_digits = 8; //sizeof(unsigned)*8;


    //Each thread use local_bucket to move data
    int64_t i;
    for(int32_t shift = 0; shift < total_digits; shift+=BASE_BITS) {
        uint32_t bucket[BASE] = {0};
 //	printf("Total digits = %d\n", total_digits);
        uint32_t local_bucket[BASE] = {0}; // size needed in each bucket/thread
        //1st pass, scan whole and check the count
        #pragma omp parallel firstprivate(local_bucket) num_threads(nthreads)
        {
            #pragma omp for schedule(static) nowait
            for(i = inf; i < sup; i++){
                local_bucket[DIGITS(data[i], shift)]++;
            }
            #pragma omp critical
            for(i = 0; i < BASE; i++) {
                bucket[i] += local_bucket[i];
            }
            #pragma omp barrier
            #pragma omp single
            for (i = 1; i < BASE; i++) {
                bucket[i] += bucket[i - 1];
            }
            uint32_t nthreads = omp_get_num_threads();
            uint32_t tid = omp_get_thread_num();
            for(int32_t cur_t = nthreads - 1; cur_t >= 0; cur_t--) {
                if(cur_t == tid) {
                    for(i = 0; i < BASE; i++) {
                        bucket[i] -= local_bucket[i];
                        local_bucket[i] = bucket[i];
                    }
                } else { //just do barrier
                    #pragma omp barrier
                }
 
            }
            #pragma omp for schedule(static)            
            for(i = inf; i < sup; i++) { //note here the end condition
                //buffer[local_bucket[DIGITS(data[i], shift)]++] = data[i];

		//uint32_t a= local_bucket[DIGITS(data[i], shift)]++;
                //buffer[a/*local_bucket[DIGITS(data[i], shift)]++*/] = data[i];            /* 1. COOOOMMENTTTTT*/ //For sorting data instead of indexes
                outIndexes[local_bucket[DIGITS(data[i], shift)]++] = Ind[i];
            }
        }
        //now move data
      /*  unsigned* tmp = data;
        data = buffer;
        buffer = tmp;*/

        
    }
	//for(size_t i=0;i<n;i++)		printf("Inside Indices [%u] %u\n ", i, outIndexes[i]);
	//printf(" %d ", arr1[i]);
	
//    Ind = outIndexes;
    memcpy ((Ind+inf), outIndexes, (sup-inf)<<2);
    //free(buffer);
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


