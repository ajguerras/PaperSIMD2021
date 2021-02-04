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

void Seq_lsd_radix_sort_(int32_t n, uint8_t *data, uint32_t *Indexes, int inf, int sup);
void gen_data(uint8_t *data, int32_t size);
void gen_Indexes(uint32_t *array, int32_t size);
int SelectValidCharsMsc (uint8_t *InputText, int32_t size, int Iter, uint32_t *Indexes, uint8_t *OutInt, int32_t *frontera, uint8_t *HU, uint32_t *ID);
void insertionSort(uint8_t *y, uint32_t *Indexes, int32_t inf, int32_t sup);
void procesar(int32_t size, uint8_t *cadena, uint32_t *Indexes, uint8_t *OutInt, int32_t *frontera, uint8_t *HU, uint32_t *ID);
int main(int argc, char* argv[]){
	
	struct timeval t1, t2;
  	double elapsedTime;	

	int32_t size;
	size = atoi(argv[1]);
	uint8_t *cadena = (uint8_t *)calloc(size+1, sizeof(uint8_t));
	if (cadena  == NULL) {
	/* Error al intentar reservar memoria */
		printf("error cadena\n");
	}
	uint32_t *Indexes = (uint32_t*) calloc(size, sizeof(uint32_t));//uint32
	if (Indexes  == NULL) {
	/* Error al intentar reservar memoria */
		printf("error Indexes\n");
	}
	uint8_t *OutInt = (uint8_t *)calloc(size, sizeof(uint8_t));
	if (OutInt  == NULL) {
	/* Error al intentar reservar memoria */
		printf("error OutInt\n");
	}
	uint32_t i;
	int32_t *frontera = (int32_t *)calloc(size, sizeof(int32_t));//32
	if (frontera  == NULL) {
	/* Error al intentar reservar memoria */
		printf("error frontera\n");
	}
	uint8_t *HU =(uint8_t*)calloc(size, sizeof(uint8_t));
	if (HU  == NULL) {
	/* Error al intentar reservar memoria */
		printf("error HU\n");
	}
	uint32_t *ID =(uint32_t*)calloc(size, sizeof(uint32_t));/*arreglo para identificar los subgrupos que van apareciendo*/
	if (ID  == NULL) {
	/* Error al intentar reservar memoria */
		printf("error ID\n");
	}
	frontera[0] = -1;
	gen_data(cadena, size);
	gen_Indexes(Indexes, size);
	//strcpy(cadena,"dccdcccdcdc");   //cdc   dcbdbdcabbcdcddcacaadadbcccdddbcccbdbadc
	char out0_file[]="sin_orden.dat";
	/*FILE *outfp0 = fopen(out0_file, "w");
  	if (outfp0 == NULL)
    	{
      		fprintf(stderr, "Error opening in file\n");
      		exit(1);
    	}
	for(i=0;i<size;i++){
			fprintf(outfp0,"%s\n", &cadena[Indexes[i]]);		
	}
	fclose(outfp0);*/
	gettimeofday(&t1, NULL);
	procesar(size, cadena, Indexes, OutInt, frontera, HU, ID);
	gettimeofday(&t2, NULL);
  	elapsedTime = (t2.tv_sec - t1.tv_sec) * 1000.0;      // sec to ms
  	elapsedTime += (t2.tv_usec - t1.tv_usec) / 1000.0;   // us to ms}
  	printf("Processing time: %5.3f ms\n", elapsedTime);
	

	/*for(i=0;i<size;i++){
		printf("%d")
	}*/

	/*char out_file[]="orden.dat";
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
	fclose(outfp);*/
	if(cadena){
		free(cadena);
	}
	if(OutInt){
		free(OutInt);
	}
	if(HU){
		free(HU);
	}
	if(ID){
		free(ID);
	}
	return 0;
}


void Seq_lsd_radix_sort_(int32_t n, uint8_t *data, uint32_t *Indexes, int32_t inf, int32_t sup) {
//Este codigo debe limpiarse una vez que se decida si se ejecutara en paralelo o puramente secuencial

    uint64_t *buffer = (uint64_t *) malloc(n*sizeof(uint64_t));
    if (buffer  == NULL) {
	/* Error al intentar reservar memoria */
		printf("error buffer\n");
    }
    int total_digits = sizeof(uint32_t)*8;
    int32_t i;
    int shift, cur_t;
    for(shift = 0; shift < total_digits; shift+=BASE_BITS) { //Teoricamentedebe descartarse. PROBAR HIP*
//generar versio limpia quitando sentencias OPENMP, comentarles
        int64_t bucket[BASE] = {0};
        int64_t local_bucket[BASE] = {0}; // size needed in each bucket
        //1st pass, scan whole and check the count
             for(i = inf; i <= sup; i++){                   //Construccin del histograma en cubetas privadas, No hay problemas de acceso a memoria.
                local_bucket[DIGITS(data[i], shift)]++;
	//	printf("%d\n", DIGITS(data[i], shift));
             }
             for(i = 0; i < BASE; i++) {                   //Totalizacion del Histograma en en una sola estructura de DAtos, en la version secuencia puede no es NECESARIO
                bucket[i] += local_bucket[i]; 
             }
             for (i = 1; i < BASE; i++) {		  //Prefix SUM, debe ejecutarse de forma estrictamente secuencial, No hay problemas de acceso a memoria.	
                bucket[i] += bucket[i - 1];
             }
            int nthreads = 1 ;
            int tid = 0;  
            for(cur_t = nthreads - 1; cur_t >= 0; cur_t--) {  // Cada hilo: Recorre todas las cubetas, le quita al PRsum, el valor d elo shistogramas locales. y ahora ese es el valor de la cubeta local
                if(cur_t == tid) {				//Estp se hace para que llos desplazamientos dependan del el valor del exclusive p sum, pero le quitan la cantidad de elmentos de esa
                    for(i = 0; i < BASE; i++) {			//clase que habia en esta cubeta. Es decir en el ciclo anterior se hace la suma de prefijos y en este se le da el caracter exclusivo.
                        bucket[i] -= local_bucket[i];
                        local_bucket[i] = bucket[i];
                    }
                } //EndIf
            }//End for cur_t

            for(i = inf; i <= sup; i++) { 			//Final partial sorting (placement), cuidar problemas de Acceso SIMULTANEO a BUFFER, q se modifica continuament
                buffer[local_bucket[DIGITS(data[i], shift)]++] = Indexes[i];
            }
        //now move data
        //unsigned* tmp = data;//Indexes
	uint32_t* tmp = Indexes;
        ///*Indexes*/data = buffer;
	int32_t h=0;
	for(i=inf;i<=sup;i++){
		Indexes[i] = buffer[h]; //Indexes = buffer; //memcpy //estos dos ciclos for en la version paralela deben desaparecer//
		h++;
	}
	h=0;
	for(i=inf;i<=sup;i++){
		buffer[h] = tmp[i];//buffer = tmp;/*OJO ACÁ*/
		h++;
	}
    } //EndFor shift
    if(buffer){
    	free(buffer);
    }
} //end_Seq


int32_t SelectValidCharsMsc (uint8_t *InputText, int32_t size, int32_t Iter, uint32_t *Indexes, uint8_t *OutInt, int32_t *frontera, uint8_t *HU, uint32_t *ID){
	   int32_t It;
	   int32_t j=1;  
	   /*aux= caracter valido previo*/
	   int16_t aux=0;
	   uint32_t aux_ID = ID[Indexes[0]];//se utiliza para saber el ID del sufijo antes de cambiarlo
	   uint32_t cont_ID=0;
	   if((Indexes[0]  +  Iter -1) >=size){
		aux = 0;
	   }else{
		aux = (int16_t) InputText [  Indexes[0]  +  Iter -1 ];
	   }
	   for (It=0;It<=size-1;++It){ 
		int32_t ValidChar = Indexes[It] + Iter;
      		if(!(ValidChar >= size)){
	             OutInt [It] = (uint8_t) InputText [  ValidChar ] ;  
		}else{
			OutInt[It] = 0;		//reevaluar
		}
		if(ValidChar>size /*&& aux!=0 */&& Iter>0){ /*este if puede sobrar*/
			frontera[j] = It -1;
			//HU[It] = 1; //agregada
			if(It==0){
				frontera[j] = It;
			}
			j++;
			aux = 0;
		}
		else if(((InputText [  ValidChar -1 ] != aux) && (Iter>0)) || (HU[Indexes[It]]==1)/*||((carcteres ante son iguales)&&(Grupo[sufijo actual]!=Grupo[sufijo anterio]))*/){//(caracter diferente) (no es la 1st iteracion)(sufijo ya ha sido ordenado)
			frontera[j] = It -1;
			if(It==0){
				frontera[j] = It;
			}
			j++;
			aux = (int16_t) InputText [  ValidChar -1 ];
			aux_ID = ID[Indexes[It]];
			cont_ID++;
			ID[Indexes[It]] = cont_ID;/*aux_ID guarda el valor del subgrupo antes de actualizarlo, para compararlo con los sufijos siguientes */
			if(HU[Indexes[It]]==1){
				aux=(-1);
			}
			
		}else if(((InputText [  ValidChar -1 ] == aux) && (Iter>0))){
			if(aux_ID==ID[Indexes[It]]){
				ID[Indexes[It]] = cont_ID;
			}else{
				frontera[j] = It -1;
				if(It==0){
					frontera[j] = It;
				}
				j++;
				aux = (int16_t) InputText [  ValidChar -1 ];
				aux_ID = ID[Indexes[It]];
				cont_ID++;
				ID[Indexes[It]] = cont_ID; 
				if(HU[Indexes[It]]==1){
					aux=(-1);
				}	
			}
		}
   
 	   }
	return j;    
}//end_SelectValidChars

void gen_data(uint8_t *data, int32_t size){
	uint32_t i;
	char num;
	//srand(time(NULL));
	for (i=0;i<size;i++){
		num = (char)((rand()%4) + 97);//desde 65 a 90
		//printf("--%d", num);
		data[i] = num;
	}
	data[size] = '\0';
	//printf("%s\n", array);		
}
void gen_Indexes(uint32_t *array, int32_t size){
	uint32_t i;
	for (i=0;i<size;i++){
		array[i] = i;
	}
}

void insertionSort(uint8_t *y, uint32_t *Indexes, int32_t inf, int32_t sup){
	int32_t i, j, key_index;
	uint8_t key;
	//printf("size: %d\n", sup);
   for (i = inf+1; i < sup+1; i++)
   {
       key = y[i];
       key_index = Indexes[i];
	//printf("  key: %d", key);
       j = i-1;
       while (j >= inf && y[j] > key)
       {
           y[j+1] = y[j];
	   Indexes[j+1] = Indexes[j];
           j = j-1;
       }
       y[j+1] = key;
       Indexes[j+1] = key_index;
   }
}

void procesar(int32_t size, uint8_t *cadena, uint32_t *Indexes, uint8_t *OutInt, int32_t *frontera, uint8_t *HU, uint32_t *ID){
	uint32_t i , x;
	int32_t subgrupos;
	uint32_t Iter=0;
	uint8_t band=1;
	char out_file[]="out-step.dat";
	/*FILE *outfp = fopen(out_file, "w");
  	if (outfp == NULL)
    	{
      		fprintf(stderr, "Error opening in file\n");
      		exit(1);
    	}
	fprintf(outfp,"\n\n\t \t ORIGINAL  \n\n");
	for(i=0;i<size;i++){
		fprintf(outfp,"\t \tsufijo # %i pos: %i \n",i, Indexes[i]);
		fprintf(outfp,"%s\n", &cadena[Indexes[i]]);		
	}*/
	do{
		subgrupos = SelectValidCharsMsc(cadena, size, Iter, Indexes, OutInt, frontera, HU, ID);
		band=1;
		/*if((frontera[0]-0) >0){
			if((frontera[0]-0) <=125){
				insertionSort(OutInt, Indexes, 0, frontera[0]);
			}else{
				Seq_lsd_radix_sort_(size,OutInt, &Indexes[0], 0, frontera[0]);
			}
		}else{
			//HU[Indexes[0]]=1;
			HU[Indexes[frontera[0]]]=1;
		}
		//band = band && HU[Indexes[0]];
		band = band && HU[Indexes[frontera[0]]];*/
		for(x=0;x<subgrupos-1;x++){
			if ( (frontera[x+1]-(frontera[x]+1)>0) ){
				if ((frontera[x+1]-frontera[x]+1<= 125)){
					insertionSort(OutInt, Indexes, frontera[x]+1, frontera[x+1]);
				}else{
					Seq_lsd_radix_sort_(size, OutInt, Indexes, frontera[x]+1, frontera[x+1]);
				}
			}else {
				HU[Indexes[frontera[x+1]]]=1;
				//HU[Indexes[frontera[x]+1]]=1; //frontera[x] ????
			}
			band = band && HU[Indexes[frontera[x+1]]];
		}
		if(subgrupos>0){ //este if sobra
			if((size-1-(frontera[subgrupos-1]+1))>0){
				if((size-1-(frontera[subgrupos-1]+1))<=125){
					insertionSort(OutInt, Indexes, frontera[subgrupos-1]+1 , size-1);
				}else{
					Seq_lsd_radix_sort_(size, OutInt, Indexes, frontera[subgrupos-1]+1 , size-1);
				}
			}else{
				HU[Indexes[frontera[subgrupos-1]+1]]=1;
				//HU[Indexes[size-1]]=1;
		
			}
			//band = band && HU[Indexes[size-1]];
			band = band && HU[Indexes[frontera[subgrupos-1]+1]];
		}
		/*fprintf(outfp,"\n\n\t \t ITERACIÓN # %i \n",Iter);
		for(i=0;i<size;i++){
			fprintf(outfp,"\t \tsufijo # %i pos: %i \n",i, Indexes[i]);
			fprintf(outfp,"%s\n", &cadena[Indexes[i]]);		
		}
		for(i=0;i<subgrupos;i++){
			fprintf(outfp,"\t \tfrontera [%i] = %i \n",i, frontera[i]);		
		}
		for(i=0;i<size;i++){
			fprintf(outfp,"\t \tHU[%i] = %i \n",i, HU[i]);		
		}*/
		Iter++;
	}while (!band);
	//fclose(outfp);
	printf("%d\n", Iter);
	/*for(i=0;i<size;i++){
		printf("%d", HU[i]);
	}*/
}
