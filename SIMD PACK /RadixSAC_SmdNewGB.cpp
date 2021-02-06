//SIMD Opt. Version 2019. Con especificaciones para prueba incorporadas al programa principal. Compilar usando make en el mismo directorio
// Previo a la compilacion debe ejecutarse: source /opt/intel/bin/compilervars.sh intel64


//KMP_AFFINITY=compact ./RadixSAC_Smd_Omp 88 88
//OMP_PROC_BIND=true

#include <iostream>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <inttypes.h>
#include <string.h>
#include <immintrin.h>
#include <chrono>




//Constants
#define MRL 16  				//Maximu register length
#define BASE_BITS 8				//Bits base para el radixsort
#define MASK (BASE-1)
#define ENDCHAR 0
#define STEP 2					//Tamano del salto orginado por consecuencia del packing
#define CARDINALIDAD 32//20000 //24000//32000// //Cardinalidad de grupos para ordenamiento secuencial
#define CARD_TLP_ONLY  				//Cardinalidad de grupos para el caso TLP
//#define SIZEHIST 66018 			//Para packing de a dos elementos
						//Importantes para modificar con el packing
#define PACK_2  				//Para definir el packing, 2 caracteres se procesan juntos

#ifdef PACK_2
	#define SIZEHIST 16273//+1+MRL//65535+MRL//() //
	#define PASOITER 2
	#define HIGHESTCHAR 16255//66018 //
	#define FACTORPACK 128//256 //
	//__m512i MASKFAKE_AVX512_256   = _mm512_set_epi32(0x101C0, 0x101C0, 0x101C0,0x101C0,0x101C0,0x101C0,0x101C0,0x101C0,0x101C0, 0x101C0, 0x101C0,0x101C0,0x101C0,0x101C0,0x101C0,0x101C0); 
	__m512i MASKFAKE_AVX512   = _mm512_set_epi32(0x4090, 0x4090, 0x4090,0x4090,0x4090,0x4090,0x4090,0x4090,0x4090, 0x4090, 0x4090,0x4090,0x4090,0x4090,0x4090,0x4090); 

#else
	#define SIZEHIST (256+MRL) //Tamaño del histograma Sin packing (16400)
	#define PASOITER 1
	#define HIGHESTCHAR (256+MRL-1) //OJO ESTE UNO LO COLOQUE EL 23-10 por problemas en el Hist.
	#define FACTORPACK 0
	__m512i MASKFAKE_AVX512   = _mm512_set_epi32(0xFF, 0xFF, 0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF, 0xFF, 0xFF,0xFF,0xFF,0xFF,0xFF,0xFF); 
#endif


//Macros
#define BASE (1 << BASE_BITS)
#define DIGITS(v, shift) (((v) >> shift) & MASK)
#define MIN(X, Y) (((X) < (Y)) ? (X) : (Y))

//Activate Insersort
//#define INSERTSORT
//#define RANGE_ISORT MRL+MRL+1

//Debug constants
#define DEBUG                     //Para generar la salida Out con los sufijos ordenados .deberia ser detallada esta salida pero se modifico la impresion
//#define DEBUG_GEN		    //Para imprimir el input aleatorio en archivo		
#define DEBUG_IS		    //Para depurar la generacion de sufijos de entrada (InputSuffixes)
#define DEBUG_H			    //Para depurar la generacion del histograma detalladamente
#define DEBUG_PS		    //Para depurar la generacion del PrefixSum
#define DEBUG_SVC		    //Para depurar Select Valid Char
#define DEBUG_PR		    //Para depurar El partial radixsort
//#define DEBUG_H_SIMD		    //Para depurar la generacion del histograma a nivel de instrucciones SIMD
#define DEBUG_MEMC		    //Para depurar globalmente el efecto del memcpy	
//#define DEBUG_SAC 		    //Para generar la salida completa del archivo, solo sufijos.
//#define DEBUG_per_ITER
//#define PRINT_PR
//#define DEBUG_EXPAND 		    //Para depurar en modo BB (Black Box) el Expand sencillo
//#define DEBUG_EXPAND_SIMD	    //Para depurar en modo WB (White Box) el Expand + Packaging
#define DEBUG_PRH 		     //Para depurar el Histograma Tipo caja negra.

//#define TESTS_MT //For automatic tests (varias replicas por experimento) 
			
			



/*******************************   Constant AVX_512 Registers    **********************************/
   __m512i MASKZERO_AVX512  = _mm512_set_epi32(0x0, 0x0, 0x0,0x0,0x0,0x0,0x0,0x0,0x0, 0x0, 0x0,0x0,0x0,0x0,0x0,0x0);
   __m512i MASKONE_AVX512   = _mm512_set_epi32(0x1, 0x1, 0x1,0x1,0x1,0x1,0x1,0x1,0x1, 0x1, 0x1,0x1,0x1,0x1,0x1,0x1);
   __m512i MASK_AVX512_1to16 = _mm512_set_epi32(16,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1);          //Used  for Initializing Suffixes and SVC
   __m512i MASK_AVX512_16    = _mm512_set_epi32(16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16); //Used  for Initializing Suffixes and SVC
   __m512i MASK_AVX512_0to15 = _mm512_set_epi32(15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0);   


   __m512i MASK_AVX512_SHIFTL_INT  = _mm512_set_epi32(14,13,12,11,10,9,8,7,6,5,4,3,2,1,0,0);
   __m512i MASK_AVX512_SHIFTL2_INT = _mm512_set_epi32(13,12,11,10,9,8,7,6,5,4,3,2,1,0,0,0);
   __m512i MASK_AVX512_SHIFTL4_INT = _mm512_set_epi32(11,10,9,8,7,6,5,4,3,2,1,0,0,0,0,0);
   __m512i MASK_AVX512_SHIFTL8_INT = _mm512_set_epi32(7,6,5,4,3,2,1,0,0,0,0,0,0,0,0,0);

   __mmask16 mask1= 0xFFFF;            //For calculating VLU

   //For popCont, eliminar innecesarias luego de decidir modelo adecuado
   __m512i low_mask = _mm512_set1_epi32 (0xF) ;
   __m512i lookup   = _mm512_set_epi8( 4,3,3,2,3,2,2,1,3,2,2,1,2,1,1,0,  4,3,3,2,3,2,2,1,3,2,2,1,2,1,1,0,    
                                       4,3,3,2,3,2,2,1,3,2,2,1,2,1,1,0,  4,3,3,2,3,2,2,1,3,2,2,1,2,1,1,0);
   __m512i V0_3, V4_7, V8_11, V12_15, cnt1_, cnt2_,cnt3_,cnt4_,sum_1, sum_2; 

   __m512i ShufflePsum  = _mm512_set_epi8( 63,62,61,60,  63,62,61,60, 63,62,61,60, 63,62,61,60, /*til here*/63,61,61,60, 63,62,61,60, 63,62,61,60, 63,62,61,60 ,63,62,61,60, 63,62,61,60, 63,62,61,60, 63,62,61,60     ,63,62,61,60, 63,62,61,60,  63,62,61,60, 63,62,61,60 );


/*******************************PROTOTYPES  **********************************/
//SEQUENTIAL
void gen_data(uint8_t *data, uint32_t size);
void gen_Indexes(uint32_t *array, uint32_t size);
void DebugHistogram(uint32_t *Hist, uint32_t *NewSortedSuf, uint32_t *Input, int64_t Inf, int64_t Sup);
void Debug_MemC(uint32_t *Suffixes, uint32_t *NewSortedSuf, int64_t Inf, int64_t Sup);
uint32_t GroupBuilder (uint32_t *InputText, uint32_t size, uint32_t Iter, uint32_t *Indexes, uint32_t *OutInt, int64_t *frontera, uint8_t *HU, uint32_t *ID);
void insertionSort(uint32_t *y, uint32_t *Indexes, uint32_t inf, uint32_t sup);
void SAtoFile(uint8_t *InputOriginal, uint32_t *Suffixes, uint32_t sizeInput );
void insertionSort(uint32_t *data, uint32_t *Indexes, int32_t inf, int32_t sup);
uint32_t SelectValidCharsMsc (uint8_t *InputText, uint32_t size, uint32_t Iter, uint32_t *Indexes, uint8_t *OutInt, int64_t *frontera, uint8_t *HU, uint32_t *ID);
void swapPointers(uint32_t **Suffixes, uint32_t **NewSortedSuf);
void ArraystoFile(uint32_t *InputOriginal, uint32_t *Suffixes, uint32_t sizeInput );

//DLP_ONLY
void print_vector(void *vec_ptr, std::string comment, std::string simdtype, std::string datatype, std::string format) ;
__m512i inline  __attribute__((always_inline)) VPIgenerate(__m512i In); 
static inline __attribute__((always_inline)) __m512i _mm512_ExPrefixSum_epi32(__m512i val) ; 
inline  __m512i VLUgenerate(__m512i Conf);
inline  __mmask16 VLUgenerateMASK(__m512i Conf);
void inline  ExcPsum (uint32_t *Hist, uint32_t *ExPsum);
void inline  Psum (uint32_t *Hist, uint32_t *ExPsum);
static inline __attribute__((always_inline)) __m512i _mm512_PrefixSum_epi32(__m512i val);
void inline  InitializeSuffixSIMD (uint32_t SizeInput, uint32_t *Suffixes);
void inline  Expand_SIMD (uint8_t *InputArr8, uint32_t *InputArr32 , uint32_t sizeInArr8); //SizeInArr8  is the size of original input  8 bits array, no padding 
void inline  Expand_Pck2 (uint8_t *InputArr8, uint32_t *InputArr32 , uint32_t sizeInArr8);


void inline  GetHistogramSIMD (uint32_t *InputArr, uint32_t sizeInArr, uint32_t *Hist, int64_t Inf, int64_t Sup,uint32_t *aux_VPI, uint16_t *aux_VLU);
void inline  SelectValidCharSIMD (uint32_t *InputText, uint32_t Size, uint32_t currentIter, uint32_t *Indexes, uint32_t *OutInt, int64_t Inf, int64_t Sup );
void inline  PartialRadixSA_SIMD (uint32_t *InputArr, uint32_t sizeInArr, uint32_t *ExPsum, uint32_t *SortedArr , uint32_t *NewSortedArr, int64_t Inf, int64_t Sup, uint32_t *In,uint32_t *aux_VPI, uint16_t *aux_VLU);
void inline  FirstPartialRadixSA_SIMD (uint32_t *InputArr, uint32_t sizeInArr, uint32_t *ExPsum, uint32_t *SortedArr , uint32_t *NewSortedArr, int64_t Inf, int64_t Sup, uint32_t *In, uint32_t *aux_VPI, uint16_t *aux_VLU );

//TLP_ONLY
void omp_lsd_radix_sort(size_t n, unsigned data[n], uint32_t *Indexes, int64_t inf, int64_t sup);

//DLP+TLP
void inline  Radix_SAC_512 (uint32_t SizeInput,  uint8_t *InputOriginal, uint32_t *Suffixes);
void inline  OPTPartialRadixSAC_SIMD (uint32_t *InputArr, uint32_t sizeInArr, uint32_t *Hist_, int64_t Inf, int64_t Sup, int Nthreads, uint32_t *In /*putExp*/, uint32_t *SortedArr , uint32_t *NewSortedArr,uint32_t *ExPsum);


/**************************************MAIN****************************************/
int main(int argc, char* argv[]){

using namespace std;
using namespace std::chrono;
using timer = std::chrono::high_resolution_clock;

	uint32_t sizeInput = (uint32_t) atol(argv[1]);
	
	//sizeInput=101;//67;//32543;//50428182;//1892;//1016732543;//;//101673;//2543;//704281892;//1083792832;//2;//33120193;//222193//333120193;//
	uint8_t *InputOriginal; 
	uint32_t *Suffixes, NextSize; 

	/****Automated TEsts Only***********/   
	#ifdef TESTS_MT
		ofstream myfile;
		myfile.open ("SAC SIMD PACK  Experiments Oct_25.txt",ios::app);
		myfile << " --------------------New Round of experiments OPT MIMD ONLY----------------------\n" << endl;
		myfile.close();	
		uint32_t NthreadsArr[6]={/*262144,*/1048576,4194304,16777216,67108864,268435456,1016732528};//11,1
		//uint32_t NthreadsArr[6]={1003,41943,167772,671088,2684354,10167325};//11,1
		int len = sizeof(NthreadsArr) / sizeof(uint32_t);

	for (int replica=0 ; replica<5; replica++)
	for (int i=0 ; i<len; i++){ 	//for (int i=len-1 ; i>=0; i--){
		sizeInput=NthreadsArr[i];///printf ("tamaño %u \n", sizeInput); fflush(stdout);
	#endif

	printf(" Entrada %u \n ", sizeInput); 
	
        int succP= posix_memalign ((void **) &InputOriginal, 64, (sizeInput+16)*sizeof(uint8_t));
	if (succP!=0){         printf ("Error reserving memory for InputOriginal");            exit(0);    }  

	
    	succP= posix_memalign ((void **) &Suffixes, 64, (sizeInput+16)*sizeof(uint32_t)); 
        if (succP!=0){         printf ("Error reserving memory for Suffixes");            exit(0);    } 

	gen_data(InputOriginal, sizeInput);


	auto startLocal = std::chrono::high_resolution_clock::now();   
	Radix_SAC_512 (sizeInput,  InputOriginal, Suffixes );
	auto stopLocal = std::chrono::high_resolution_clock::now();   
	
	#ifdef TESTS_MT
	
		cout << replica<< " Replica, "<< (uint32_t )sizeInput << " Sufijos Ordenados,  Milliseconds in PACKSIMD RadixSAC: "<< duration_cast<milliseconds>(stopLocal-startLocal).count() << ".\n" << endl;
		myfile.open ("SAC SIMD PACK  Experiments Oct_25.txt",ios::app);
		myfile <<  replica<< " Replica, "<< (uint32_t )sizeInput<< " Sufijos Ordenados,  Milliseconds in PACKSIMD RadixSAC:"<< duration_cast<milliseconds>(stopLocal-startLocal).count() << ".\n" << endl;
		myfile.close();	
        //cout << (uint32_t) NextSize<<" Hilos "<< duration_cast<milliseconds>(stopLocal-startLocal).count() << " Chronos milliseconds in MIMDRadix (cout).\n" << endl;

	#ifdef DEBUG_SAC
		SAtoFile(InputOriginal, Suffixes, sizeInput );
		printf ("El archivo ha sido generado pulse para continuar la proxima execucion"); int c=getchar();
	#endif
	}
	#endif
	
	/************************************End of automated tests*****************************************/


	#ifdef DEBUG  
		int a=0;  char SizeInString[36];
		sprintf(SizeInString, "OutSACOPT512_%d.txt", sizeInput);
	
	 	FILE *outFile=fopen (SizeInString, "w"); 
	   	//fprintf (outFile, "Starting Radixsort... \n ");fflush(stdout);
		//fprintf (outFile, "\n %lld Milliseconds in RADIXSIMD.\n",  duration_cast<milliseconds>(stopLocal-startLocal ).count()) ;*/
		
	     	for (int32_t i=0; i<sizeInput;++i){ 
			unsigned char *menor, *mayor;
		 	//Formato para analisis visual de resultados.	
		 	//fprintf (outFile,"Suf# %i , FinalPos %d = %s\n",Suffixes[i],i, InputOriginal+Suffixes[i]) ; //If  you want to see all the suffixes in a file

			//Formato para verificador de salida de Juan David
			fprintf (outFile,"%s\n",InputOriginal+Suffixes[i]) ; //If  you want to see all the suffixes in a file
		/*	menor=(unsigned char*)InputOriginal+Suffixes[i] ;  mayor=(unsigned char*)InputOriginal+Suffixes[i+1]; 
			a =strcmp((char *)menor, (char *)mayor);      
		 	if  ((i<(sizeInput-1))&&((*menor>*mayor)||(a>0))) {
			    fprintf(outFile,"\n ****ERROR GRAVE EN EL ORDENAMIENTO FINAL POS %i  \niMenor: %s \nMayor : %s\n ",i, menor , mayor); fflush(stdout); 
				
			    printf("\n ****ERROR GRAVE EN EL ORDENAMIENTO FINAL POS %i  menor: %s\n mayor : %s\n ",i, menor , mayor); fflush(stdout);   
			   int c = getchar ();               	
			}*/
	      }
	      if (outFile) fclose (outFile);
	#endif

	//printf ("ANTES DE LIBERAR MEMORIA PARA TERMINAR\n");fflush(stdout); 
	if(InputOriginal)		free(InputOriginal);  //SUBIR y colocar luego del expand
	if(Suffixes)			free(Suffixes);
	
	return 0;
}




//En el futuro deb eintervenirse para genrear version con paralelismo de grano grueso. (x grupos)
void inline  Radix_SAC_512 (uint32_t sizeInput,  uint8_t *InputOriginal, uint32_t *Suffixes ){


  	double elapsedTime;
	int64_t totalcard=0;
	uint32_t sizeInputPad=sizeInput+16; //Input Size Padded

	uint32_t sizeH=SIZEHIST*sizeof(uint32_t);

        uint32_t sizeInputMEM=sizeInput*sizeof(uint32_t); //Sizes for memory allocation 
        uint32_t sizeInputMEMPad=sizeInputPad*sizeof(uint32_t); //Sizes for memory allocation Padded

    	uint32_t *InputExp; 
        int succP= posix_memalign ((void **) &InputExp, 64, sizeInputMEMPad);
	if (succP!=0){         printf ("Error reserving memory for InputEx");            exit(0);    }  

	#ifdef INSERTSORT
	int64_t i_, j_ ;
	uint32_t key, key_index, *auxInputIns;
	succP= posix_memalign ((void **) &auxInputIns, 64, RANGE_ISORT<<2); //34
	#endif

	uint32_t *OutChar; //Current chars to be evaluated in each iteration
    	succP= posix_memalign ((void **) &OutChar, 64, sizeInputMEMPad);
	if (succP!=0){        printf ("Error reserving memory for OutChar");           exit(0);    }  

	uint32_t *Hist; //Histogram,   
    	succP= posix_memalign ((void **) &Hist, 64, sizeH);
	if (succP!=0){        printf ("Error reserving memory for Histogram");       	exit(0);    }  

	uint32_t *ExPsum; //Exclusive PrefixSum
	succP= posix_memalign ((void **) &ExPsum, 64, sizeH);  
	if (succP!=0){         printf ("Error reserving memory for ExPsum");              exit(0);    }

	uint32_t *NewSortedSuf; //For sorting Suffixes
	int succP_= posix_memalign ((void **) &NewSortedSuf, 64, sizeInputMEMPad);
	if (succP_!=0){         printf ("Error reserving memory for NewSortedArray");            exit(0);    }  

	int64_t *Frontera; //uint32_t *Frontera; 
	//succP= posix_memalign ((void **) &Frontera, 64, sizeInputMEMPad);
        succP= posix_memalign ((void **) &Frontera, 64, (sizeInputPad)*sizeof(int64_t));
	if (succP!=0){         printf ("Error reserving memory for Frontera");            exit(0);    } 

	uint32_t *ID; 
        succP= posix_memalign ((void **) &ID, 64, sizeInputMEMPad);
	if (succP!=0){         printf ("Error reserving memory for ID");            exit(0);    } 

	uint8_t *HU; 
        succP= posix_memalign ((void **) &HU, 64, (sizeInputPad)*sizeof(uint8_t));
	if (succP!=0){         printf ("Error reserving memory for HU");            exit(0);    } 

	uint32_t *aux_VPI;
	succP= posix_memalign ((void **) &aux_VPI, 64, sizeInputMEMPad);
	if (succP!=0){        printf ("Error reserving memory for aux_VPI");           exit(0);    }   

	uint16_t *aux_VLU;
	succP= posix_memalign ((void **) &aux_VLU, 64, (sizeInputMEMPad>>5));//4 por la division entre 16, 1 porque este es el size de 32
	if (succP!=0){        printf ("Error reserving memory for aux_VLU");           exit(0);    }  

	InitializeSuffixSIMD (sizeInput, Suffixes);

	memset(Frontera,0,sizeInputMEMPad);            memset(ID,0,sizeInputMEMPad);      
	//memset(OutChar,0,sizeInputMEMPad);              //memset(NewSortedSuf,0,sizeInputMEMPad);
	memset(HU,0,(sizeInputPad)*sizeof(uint8_t));          

	#ifdef PACK_2                         //THIS THE PACKING
		Expand_Pck2 (InputOriginal, InputExp , sizeInput); 	
	#else
		Expand_SIMD (InputOriginal, InputExp,  sizeInput); 
	#endif

	

	Frontera[0] = -1;
	uint32_t auxSwap,x, subgrupos=1;
	uint32_t currentIter=0;// Inf=Frontera[x]+10, Sup=sizeInput (Frontera[x+1];
	uint8_t band=1;
	int64_t Inferior, Cardinalidad, aux = (int64_t)sizeInput-1;//, groupCounter=0, groupHugeCounter=0;

	memset(Hist,0,sizeH); memset(ExPsum,0,sizeH); 
	
	
	/************************************Begin VERSIONING ************************************/	
	GetHistogramSIMD (/*OutChar*/ InputExp, sizeInput, Hist, 0,  sizeInput,aux_VPI,aux_VLU);
	//for (int m=0 ; m< 16; m++) printf (" Hist [%i]=%u ", m,Hist[m]);fflush(stdout);
	
	ExcPsum (Hist, ExPsum);	//printf("\t ExPSUM ");fflush(stdout);	
	//for (int m=0 ; m< 16; m++) printf (" ExPs [%i]=%u ", m,ExPsum[m]);fflush(stdout); int c=getchar();
	//int c=getchar();
	//ArraystoFile(Suffixes, NewSortedSuf,  sizeInput); c=getchar();
	PartialRadixSA_SIMD (/*OutChar*/InputExp, sizeInput, ExPsum, Suffixes, NewSortedSuf, 0,  sizeInput,InputExp,aux_VPI,aux_VLU); //int d = getchar();
	//printf ("after exp ");fflush(stdout);
	//printf ("PARTIAL DONE");
	//ArraystoFile(Suffixes, NewSortedSuf,  sizeInput);/*int*/ c=getchar();
	//swapPointers(&Suffixes, &NewSortedSuf); //_mm512_store_epi32 (Suffixes+sizeInput,MASKZERO_AVX512); 
	//ArraystoFile(Suffixes, NewSortedSuf,   sizeInput); c=getchar();
	//SAtoFile(InputOriginal, NewSortedSuf/*Suffixes*/, sizeInput );int c=getchar();
	

	memcpy(Suffixes, NewSortedSuf, (sizeInput<<2) );//x4 porque cada dato ocupa 4
	//SAtoFile(InputOriginal, Suffixes, sizeInput );
	//ArraystoFile(Suffixes, NewSortedSuf,  sizeInput); int c=getchar();

	#ifdef DEBUG_PRH 
		DebugHistogram(Hist, NewSortedSuf, InputExp, 0, sizeInput );
		printf("First version Histogram test passed\n"); int l=getchar();
	#endif 
	#ifdef DEBUG_MEMC
		Debug_MemC(Suffixes, NewSortedSuf, 0, sizeInput);
		printf("First version MECPY test passed\n"); int m=getchar();
	#endif 
	
	currentIter+=PASOITER;	
	subgrupos = GroupBuilder (InputExp, sizeInput, currentIter, Suffixes, OutChar, Frontera, HU, ID);

	/************************************END Functions VERSIONING ************************************/	
	band=1; 
	do{//Aqui comienza el ciclo que produce el orden
		
		Frontera [subgrupos] = aux ; /*OJO cuidar*/
		band=1; //printf ("START"); fflush(stdout);

	
		for(x=0;x<subgrupos;x++){
			Cardinalidad= (int64_t) Frontera[x+1]-Frontera[x];
			if ( Cardinalidad>1 /*((MRL+1)*Nthreads)*/ ){ //Intervenir  y meter el Insersort para menos de 130 elementos.

				//printf ("ENTERing \n");fflush(stdout);
				memset(Hist,0,sizeH);  memset(ExPsum,0,sizeH); 
				Inferior=Frontera[x]+1;

				if (Cardinalidad> 2/*(CARDINALIDAD)*/){

					SelectValidCharSIMD (InputExp, sizeInput, currentIter, Suffixes, OutChar, Inferior, Frontera[x+1]+1);
					//printf ("SVC done");fflush(stdout);
					GetHistogramSIMD (OutChar, sizeInput, Hist, Inferior, Frontera[x+1]+1,aux_VPI,aux_VLU);
					//printf ("HIST done");fflush(stdout);
					ExcPsum (Hist, ExPsum);	//printf("\t ExPSUM ");fflush(stdout);	
					//printf ("EPS done");fflush(stdout);
					PartialRadixSA_SIMD (OutChar, sizeInput, ExPsum, Suffixes, NewSortedSuf, Inferior, Frontera[x+1]+1,InputExp,aux_VPI,aux_VLU);
					//printf ("PARTIAL DONE");

					memcpy(Suffixes+Inferior, NewSortedSuf+Inferior, (Cardinalidad<<2) );//Estoy asumiendo orden por grupos  yNewSorted Global 	
				}else{    
					//if (Cardinalidad> 2){

					/*		for (i_ = 0; i_ <  Cardinalidad; i_++) {
								auxInputIns[i_]=InputExp[Suffixes[i_+Inferior]];
								
							}//SVC
							//printf ("A");fflush(stdout);
							for (i_ = 0; i_ <  Cardinalidad; i_++)
							{
							       key = auxInputIns[i_];
							       key_index = Suffixes[i_+Inferior];
								//printf("  key: %d", key);
							       j_ = i_-1;
							       while ((j_ >= 0) && (auxInputIns[j_] > key))
							       {
								   auxInputIns[j_+1] = auxInputIns[j_];
								   Suffixes[j_+1] = Suffixes[j_+Inferior];
								   j_ = j_-1;
							       }
							       auxInputIns[j_] = key;
							       Suffixes[j_+1+Inferior] = key_index;
							 }
							//printf ("B");fflush(stdout);
							for (i_ = Inferior; i_ <  Frontera[x+1]+1; i_++) HU[Suffixes[i_]]=1;//Marcar HU de todos				
							
						*/	

					//}else{ //2 elements only
						//printf ("Two Sort\n");fflush(stdout);
						/*********************************************BEGIN DIRECT SORTING***********************************************/
						if (InputExp[Suffixes[Inferior]]> InputExp[Suffixes[Frontera[x+1]]]){
							auxSwap=Suffixes[Inferior]; Suffixes[Inferior] = Suffixes[Frontera[x+1]]; Suffixes[Frontera[x+1]]=auxSwap;//Reordene los indices
							HU[Suffixes[Frontera[x+1]]]=1;HU[Suffixes[Inferior]]=1; 		//Marque HU	
						}
						/*********************************************END DIRECT SORTING***********************************************/
						//printf ("OUT OF Two Sort\n");fflush(stdout);
					//}
					//printf("A");fflush(stdout);

				}


				#ifdef DEBUG_PRH 
					DebugHistogram(Hist, NewSortedSuf, InputExp, Inferior, Frontera[x+1]+1 );
				#endif 

				#ifdef DEBUG_MEMC
					Debug_MemC(Suffixes, NewSortedSuf, Inferior, Frontera[x+1]+1);
				#endif 
				#ifdef DEBUG_per_ITER	
					SAtoFile(InputOriginal, Suffixes, sizeInput );
				#endif	

			}else {

				//if (Cardinalidad>1){
					//Call OpenMp y sicronizar estructuras de datos.
				//}else
				HU[Suffixes[Frontera[x+1]]]=1; 		//printf("\t HUPDated ");fflush(stdout);		
			}
			band = band && HU[Suffixes[Frontera[x+1]]];	//printf("\t bandUpd ");fflush(stdout);		

			//printf ("Grupo actual %u desde  %u hasta %u \n", x, Inferior, Frontera[x+1]+1);
			//if (currentIter==0) break;
		}//endfor

		currentIter+=PASOITER;
//auto startLocal1 = std::chrono::high_resolution_clock::now();  
		//printf ("Bef SGr\n");fflush(stdout);
		if (!band )subgrupos = GroupBuilder (InputExp, sizeInput, currentIter, Suffixes, OutChar, Frontera, HU, ID);//Deberia ser OutInt la entrada
		//printf ("After SGr\n");printf ("band = %d", band); fflush(stdout); 
//auto stopLocal1 = std::chrono::high_resolution_clock::now();   
//cout << duration_cast<milliseconds>(stopLocal1-startLocal1).count() << " milliseconds in GROUPBUILDER.\n" << endl;
		//if (currentIter==1) break;
		//printf("\t GBuil ");fflush(stdout);		
 		//Esto por ahora porqu eel constructor de grupo si no se puede hacer en OPENMP entonces debe ser solo con el grupo actual.
	
		
		//printf ("Iteración General %u\n", currentIter);
	}while (!band);
	//SAtoFile(InputOriginal, Suffixes, sizeInput );
	//cout <<sizeInput << "En la Entrada" << groupCounter<< "Grupos totales "<< groupHugeCounter<<" Grupos Gigantes.\n" << endl;
	
//	cout <<Nthreads<<" Hilos "<< totalcard <<" Total Card counter .\n" << endl;
	if(InputExp)			free(InputExp);
	if(Hist)			free(Hist);
	if(OutChar)			free(OutChar);
	if(ExPsum)			free(ExPsum);
	if(NewSortedSuf)		free (NewSortedSuf);
	if(Frontera)			free (Frontera);
	if(HU)				free (HU);
	if(ID)				free (ID);
	if(aux_VPI)			free (aux_VPI);
	if(aux_VLU)			free (aux_VLU);
	#ifdef INSERTSORT 
		//if (auxInputIns)			free(auxInputIns);
	#endif
}//EndRadixsortSA

/************************SIMD ONLY CODE *********************************/

void inline  ExcPsum (uint32_t *Hist, uint32_t *ExPsum){          //GLOBAL PREFIXSUM: Calculates ExPsum over the input Array, multiple of 16
         uint32_t *auxH, *aux1;
         __m512i Add, dat, LocExPSum ; 
         __m512i auxAdd = MASKZERO_AVX512;                       //Initialize the aux register that helps in calculating the global Psum

         for (int x=0; x<(SIZEHIST-MRL);x+=MRL){   
                auxH=(Hist+x);

	       dat=_mm512_load_epi32 ((void *) auxH);   	 

 	       LocExPSum= _mm512_ExPrefixSum_epi32(dat);        //Calculate localReg ExPsum
 
               Add = _mm512_add_epi32 (LocExPSum,auxAdd);	 //Add with the previous iteration Psum and this is the real general Xpsum
 		 
               _mm512_store_epi32 ((void*) (ExPsum+x),Add);

               auxAdd = _mm512_add_epi32(Add, dat); 		//Updates the auxiliarReg  for next iteration
               aux1 = (uint32_t*)&auxAdd ;  

               //La otra opcion si se quiere mantener todo en registros es :  haciendo un shuffle de la ultima posicion a todas las de AuxAdd? How?
               auxAdd = _mm512_set1_epi32(aux1[15]); //o accederlos aca directamente ?  

              #ifdef PRINT_PS
                   //cout << "ExPrefixSum Iteration " << x/16 << "de 16\n";
                   //print_vector((void*)&dat, "InputDat", "AVX512", "uint32_t", "DEC");
                   //print_vector((void*)&LocExPSum, "LocExPSum", "AVX512", "uint32_t", "DEC");    
                   //print_vector((void*)&Add, "FinalPSum", "AVX512", "uint32_t", "DEC");           
                   __m512i Test= _mm512_shuffle_epi8 (auxAdd, ShufflePsum);
  		   cout << "Acarreo del prefix Sum Ex " <<aux1[15] << " \n "; 
                   print_vector((void*)&auxAdd, "auxAdd", "AVX512", "uint32_t", "DEC");          
                   print_vector((void*)&Test,   "TestSh", "AVX512", "uint32_t", "DEC");         
 	           int c=getchar(); 
               #endif 
         }//end for x=0
       
         #ifdef DEBUG_PS
             uint32_t auXPsum[SIZEHIST];
             auXPsum[0]=0;  
             if (0!=ExPsum[0] ) printf ("Error ExPsumCalc[%d]= %d ERROR vs ExPsumDes[%d]= %u \n",0, ExPsum[0],0,auXPsum[0]) ;
             for (int x=1; x<SIZEHIST-MRL;++x){  
                auXPsum[x]=auXPsum[x-1]+Hist[x-1];
                if (auXPsum[x]!=ExPsum[x] ) {printf ("Error ExPsumCalc[%d]= %d ERROR vs ExPsumDes[%d]= %u \n",x, ExPsum[x],x,auXPsum[x]);int c=getchar(); };
             }	   
            
         #endif 

} //end of ExcPsum  





inline  __m512i VLUgenerate(__m512i Conf){
   __mmask16 mask=(__mmask16)_mm512_reduce_or_epi32(Conf);        // Builds a mask with all of the elements that are 1 o more 
   return( _mm512_mask_set1_epi32(MASKONE_AVX512, mask,0));       //Puts Zero in all of the positions previously repeated, according to 1's in the mask  
}   


inline  __mmask16 VLUgenerateMASK(__m512i Conf){
   return( (__mmask16) _mm512_reduce_or_epi32(Conf) );              
}
 
static inline __attribute__((always_inline)) __m512i _mm512_ExPrefixSum_epi32(__m512i val) {            //Register Exclusive Psum 
 
   __m512i tmp_res = _mm512_mask_permutexvar_epi32(MASKZERO_AVX512, 0xFFFE,MASK_AVX512_SHIFTL_INT,val); //Moves to right 1pos to make it exclusive
   __m512i val_shifted  = _mm512_mask_permutexvar_epi32(MASKZERO_AVX512, 0xFFFE,MASK_AVX512_SHIFTL_INT,tmp_res);
   tmp_res      = _mm512_add_epi32(tmp_res, val_shifted);
   val_shifted  = _mm512_mask_permutexvar_epi32(MASKZERO_AVX512,0xFFFC, MASK_AVX512_SHIFTL2_INT, tmp_res);
   tmp_res      = _mm512_add_epi32(tmp_res, val_shifted);
   val_shifted  = _mm512_mask_permutexvar_epi32(MASKZERO_AVX512,0xFFF0, MASK_AVX512_SHIFTL4_INT, tmp_res);
   tmp_res      = _mm512_add_epi32(tmp_res, val_shifted);
   val_shifted  = _mm512_mask_permutexvar_epi32(MASKZERO_AVX512,0xFF00, MASK_AVX512_SHIFTL8_INT, tmp_res);
   tmp_res      = _mm512_add_epi32(tmp_res, val_shifted);
   return tmp_res;
}

static inline __attribute__((always_inline)) __m512i _mm512_PrefixSum_epi32(__m512i val) {            //Register Exclusive Psum 
 
   __m512i tmp_res = val; //Moves to right 1pos to make it exclusive
   __m512i val_shifted  = _mm512_mask_permutexvar_epi32(MASKZERO_AVX512, 0xFFFE,MASK_AVX512_SHIFTL_INT,tmp_res);
   tmp_res      = _mm512_add_epi32(tmp_res, val_shifted);
   val_shifted  = _mm512_mask_permutexvar_epi32(MASKZERO_AVX512,0xFFFC, MASK_AVX512_SHIFTL2_INT, tmp_res);
   tmp_res      = _mm512_add_epi32(tmp_res, val_shifted);
   val_shifted  = _mm512_mask_permutexvar_epi32(MASKZERO_AVX512,0xFFF0, MASK_AVX512_SHIFTL4_INT, tmp_res);
   tmp_res      = _mm512_add_epi32(tmp_res, val_shifted);
   val_shifted  = _mm512_mask_permutexvar_epi32(MASKZERO_AVX512,0xFF00, MASK_AVX512_SHIFTL8_INT, tmp_res);
   tmp_res      = _mm512_add_epi32(tmp_res, val_shifted);
   return tmp_res;
}

__m512i inline __attribute__((always_inline))  VPIgenerate(__m512i In){ 

   // Separate in 4x8 bits pieces 
   V0_3 = _mm512_and_si512 (In, low_mask); 
   V4_7 = _mm512_and_si512 (_mm512_srli_epi32 (In, 4 ), low_mask) ;          
   V8_11 = _mm512_and_si512 (_mm512_srli_epi32 (In, 8 ), low_mask) ; 
   V12_15 = _mm512_and_si512 (_mm512_srli_epi32 (In, 12 ), low_mask) ; 
 
   cnt1_ = _mm512_shuffle_epi8 (lookup, V0_3);     //Count number of ones
   cnt2_ = _mm512_shuffle_epi8 (lookup, V4_7);   
   cnt3_ = _mm512_shuffle_epi8 (lookup, V8_11);   
   cnt4_ = _mm512_shuffle_epi8 (lookup, V12_15);
 
   sum_1 = _mm512_add_epi32 ( cnt1_,  cnt2_);      //Add partial results 
   sum_2 = _mm512_add_epi32 ( cnt3_,  cnt4_);

   return (_mm512_add_epi32 ( sum_1,  sum_2)) ;    //Add final results
}



/**********************************UNUSED *****************************************/

void inline  SelectValidCharSIMD (uint32_t *InputText, uint32_t Size, uint32_t currentIter, uint32_t *Indexes, uint32_t *OutInt, int64_t Inf, int64_t Sup){// Indexes = SortedSuffixes, SizeInput is exact


        	//int32_t It, Displace = (int32_t) (SizeInput-1)-(int32_t)currentIter;          , in the MSC approach is just the currentIter
	uint32_t  It; 
        __m512i Suffix, TextIndex, Chars, Displ;  
	//uint32_t *auxSuf=(Indexes+currentIter);        
	Displ = _mm512_set1_epi32(currentIter);             //Displacement value 

	
	//printf ("Inf %i Sup %i high %i",Inf, Sup, high );fflush(stdout);
        for (It=Inf;It<Sup;It+=MRL){            		 
		//printf("x");	fflush(stdout);
             Suffix = _mm512_load_epi32 ((void*) (Indexes+It)) ;                //2. Load contiguos suffixes
       
             TextIndex= _mm512_add_epi32(Suffix, Displ) ;                       //3. Sum those sufixes and the Displace Value

             Chars = _mm512_i32gather_epi32 (TextIndex, (void *) InputText, 4);    //4.- Bring chars from text 
             _mm512_store_epi32 (OutInt+It,Chars);                              //5.- Store them in the OutPut



        }//end of for It=0   
	//printf ("\nInf %i Sup %i high %i It%i ",Inf, Sup, high, It );fflush(stdout);
	//Ahora debe CONTROLARSE MANUALMENTE el remanente de SVC para no dañar valores validos por eso el ciclo superior queda en menor estricto
	uint32_t ValidChar;
	
        for (uint32_t auxIt=It;auxIt<Sup;auxIt++){   
		//printf("a");	
		ValidChar = Indexes[It] + currentIter;
		if((ValidChar < Size)){       //PAdding 
		    OutInt [It] = InputText [  ValidChar ] ;  
		}else{
		    OutInt[It] = HIGHESTCHAR;		//reevaluar, Caracter Invalido. Debe ser cero o debe ser un numero muy grande para que no afecte el prefix sum
		}
	}

       _mm512_store_epi32 ((OutInt+(Sup)),MASKFAKE_AVX512); //1 more time, for the fakes values, igher so it does not affect the prefix sum 
 


	 #ifdef DEBUG_SVC
	   for (uint32_t It=Inf;It<Sup;++It){  //Para cada uno d elos caracterres validos    

		uint32_t aux, ValidChar = Indexes[It] + currentIter;
		if(!(ValidChar >= Size)){       //PAdding 
		    aux = InputText [  ValidChar ] ;  
		}else{
		    aux = HIGHESTCHAR;		//reevaluar, Caracter Invalido. Debe ser cero o debe ser un numero muy grande para que no afecte el prefix sum
		}
		if ((OutInt[It]!=aux)&&(aux!=HIGHESTCHAR)) {
			printf("\nSVC_ERROR Iter %u Elem %"PRIu32"     Deseada=  %"PRIu32"      Calculada %"PRIu32"       ValidChar = %"PRIu32"  \n", currentIter,It, aux, OutInt[It], ValidChar  ); 
			int c = getchar ();	
		} //else printf("\nSVC Deseada=  %"PRIu32"      Calculada %"PRIu32"       ValidChar = %"PRIu32"  \n", aux, OutInt[It], ValidChar  ); //printf ("SVC OK");
	
	   }
	#endif

}



void inline  InitializeSuffixSIMD (uint32_t SizeInput, uint32_t *Suffixes){  //SizeInput is measured from zero, so when calling passes SizeInput-1

    __m512i Res=MASK_AVX512_0to15; 		  //1. Fill  register 0 - 16

   for (uint32_t x=0;x<=SizeInput; x+=MRL  ){ 

        _mm512_store_epi32 (Suffixes+x, Res);   	  //2. Store Result

	Res= _mm512_add_epi32 (Res, MASK_AVX512_16);   	  //3. Add 16 to every data 

   }
    Res=_mm512_set1_epi32(SizeInput);
   _mm512_store_epi32 (Suffixes+SizeInput, Res);        //5.FakeValues for the Remaining remaining elements  

    #ifdef DEBUG_IS
    	for (uint32_t i=0; i<SizeInput;++i)         { if (Suffixes[i]!=i){ printf ("***SuffixPos[%d]= %d\n  ***********\n",i, Suffixes[i]) ;int c=getchar();} }
      
  	/*for (int i=0; i<sizeInArrF;++i) {
            if (SPos[i]!=(sizeInArr-i-1))   { 
		printf ("********Main Error Initializing SuffixPos[%d]= %d\n  ***********",i, SPos[i]) ; 
		SPos[i]=(sizeInArr-i); 
            }
         }
       int c = getchar ();*/
    #endif

}//End Initialize SuffixSIMD



void inline  GetHistogramSIMD (uint32_t *InputArr, uint32_t sizeInArr, uint32_t *Hist, int64_t Inf, int64_t Sup, uint32_t *aux_VPI, uint16_t *aux_VLU ){

   uint32_t *auxInput;  
   __m512i data,outConf, VPI, VLU, VPI1, currentData, _Add, auxPOP;  
   __mmask16 mymask, mymask_, mymask2;
   uint32_t x, auxmask=0;
   int64_t Fin ;
   if ((Sup-Inf)>=MRL) Fin= Sup-MRL;
	else Fin= 0;

   //printf ("\nInf %u , Fin %u Sup %u \n ", Inf, Fin, Sup);
   for ( x=Inf; x<=Fin;x+=MRL){ 

        //auxInput=(InputArr+x);                
        data = _mm512_load_epi32 ((void*) (InputArr+x)) ;

        outConf = _mm512_conflict_epi32(data);        
        VPI= VPIgenerate(outConf);
        mymask = VLUgenerateMASK(outConf)^mask1 ; //VLU = VLUgenerate(outConf);

   	//  *****STORES VLU and VPI to load them when applying RadixSort************CUIDADO CON EL TEMA DE ESTAS POSICIONES ***
	_mm512_store_epi32 ((aux_VPI+x),VPI);
	aux_VLU[auxmask]=mymask; auxmask++;

        VPI1 = _mm512_add_epi32(MASKONE_AVX512,VPI);   
	
        currentData = _mm512_mask_i32gather_epi32 (MASKZERO_AVX512, mymask,data , (void const *)Hist,4); //bring indexes from GLOBALHist in memory
        //si el SAC es únicamente para genómica se pueden eliminar todas las cubetas y hacer el cálculo del histograma directamente sobre Registros
        _Add = _mm512_add_epi32 (currentData, VPI1);                         //Sum The local Histogram (just calculated) with the previously existing 
        _mm512_mask_i32scatter_epi32 ((void *)Hist, mymask,data , _Add, 4) ; //Update Global Hist, needs the mask for the one that has been added

	#ifdef DEBUG_H_SIMD         
	   printf ("********* Histogram's Hiteration***** %u, SizeInArr %d \n ", x, sizeInArr);
           printf("mymask Hexa = %x \n",mymask);   
           print_vector((void*)&data,    "Data_Index", "AVX512", "uint32_t", "DEC");
           //print_vector((void*)&outConf, "OutCnf    ", "AVX512", "uint32_t", "DEC");
           //print_vector((void*)&VLU,     "VLU       ", "AVX512", "uint32_t", "DEC");           
           print_vector((void*)&VPI,     "VPI       ", "AVX512", "uint32_t", "DEC");
           //print_vector((void*)&auxPOP,  "auxPOP    ", "AVX512", "uint32_t", "DEC");
           //print_vector((void*)&VPI1,    "VPI+1     ", "AVX512", "uint32_t", "DEC");   
           print_vector((void*)&currentData, "DatFromMem", "AVX512", "uint32_t", "DEC");
           print_vector((void*)&_Add,        "   SumHist", "AVX512", "uint32_t", "DEC");
	   for (int m=0 ; m<MRL; m++) if (Hist[InputArr[x+m]]<0) {printf (" ERROR Hist [%i]=%u ",InputArr[m+x], Hist[InputArr[x+m]]);fflush(stdout);	int c=getchar();}
	   int d=getchar();
           
        #endif
   }//end for x=0
   //printf ("\n Inf %u , x=%u Fin %u Sup %u \n ", Inf, x, Fin, Sup);
   for (uint32_t auxIt=x;auxIt<Sup;auxIt++){ 
		//printf ("soy HIST SEQ, auxIt %d Sup %d\n", auxIt, Sup);
		Hist[InputArr[auxIt]]++;
	}

   #ifdef DEBUG_H         
      //for (int32_t i=0; i<sizeInArr;++i)  if( InputArr[i]>0)printf ("Input[%d]= %d  e Hist = %d \n",i, InputArr[i], Hist[InputArr[i]]) ; /*int c=getchar();*/
      int TestH[SIZEHIST];
      for (int32_t i=0; i<SIZEHIST;++i)   TestH[i]=0;   
      for (int32_t i=Inf; i<Sup;++i)  TestH[ InputArr[i]  ]++;
      for (int32_t i=0; i<SIZEHIST;++i)   
 	    if ((TestH[i]!=Hist[i])||(Hist[i])>(Sup-Inf)||(Hist[i]<0))  { printf ("CalculatedH[%u]= %u ERROR vs DesiredHist[%u]= %u \n",i, Hist[i],i,TestH[i]) ; int c=getchar();}
   #endif

}//end of GetHistogram




//Expand unsigned char Input into uint32_t to be processed. A*256+B .-->2_8

void inline  Expand_SIMD (uint8_t *InputArr8, uint32_t *InputArr32 , uint32_t sizeInArr8){ //SizeInArr8  is the size of original input  8 bits array, no padding 

   __m128i load128;  
   __m512i load512, High = _mm512_set1_epi32(255);       //1. Fill  register Size  with the exact input size



   for (uint32_t i=0;i<sizeInArr8;i+=MRL ){ 

        load128 = _mm_loadu_si128((__m128i *)(InputArr8+i));   	//1. Sequential Load,  8 bit int 
        load512 = _mm512_cvtepu8_epi32 (load128);   		//2. Expand to 32 bits
 
        _mm512_store_epi32 (InputArr32+i,load512);   		//4. Store in new array.
   }
   _mm512_store_epi32 ((InputArr32+(sizeInArr8)), High);            //Put 16 Higher al final de salida, 3 se considera el excedente *******Quizas no sea necesario.****


    #ifdef DEBUG_EXPAND
	/*print_vector((void*)&In64_8,"     In64x8", "AVX512", "uint8_t", "DEC");printf("\n");
	print_vector((void*)&Thous,"       Thou", "AVX512", "uint32_t", "DEC");
	print_vector((void*)&Hunds,"       Hund", "AVX512", "uint32_t", "DEC");
	print_vector((void*)&Tens, "       Tens", "AVX512", "uint32_t", "DEC");
	print_vector((void*)&Units,"       Units", "AVX512", "uint32_t", "DEC");

	//print_vector((void*)&MuThs,"    MuThou2", "AVX512", "uint32_t", "DEC");      		
        //print_vector((void*)&MuHs, "     MuHund", "AVX512", "uint32_t", "DEC");
        //print_vector((void*)&Mutns,"     MuTens", "AVX512", "uint32_t", "DEC");

      	print_vector((void*)&Add,  "   Final Add", "AVX512", "uint32_t", "DEC");
	//printf("\n");

        */
   	//print_vector((void*)&Tens,    "Packed Values ", "AVX512", "uint32_t", "DEC");i
         int32_t aux8=0;
         for (uint32_t y=0;y<sizeInArr8;y++ ){ 
                      //aux8 =  (InputArr8[y]*256)+(InputArr8[y+1]) ;
                      //printf(" Elements[%d]: %d, %d, %d, %d \n",y, InputArr8[y],InputArr8[y+1],InputArr8[y+2],InputArr8[y+3] );
                      if (InputArr8[y] !=InputArr32[y]) {  
                           printf(" Expand Error Output32[%d]: %d  Vs %d (8 bits) ,  In1 %u In2 %u  \n",y, InputArr32[y], aux8 , InputArr8[y],InputArr8[y+1]); fflush(stdout);
		  	   int c=getchar();	
		      }
         }
     //for (uint32_t y=0;y<sizeInArr8;y++ ) printf(" Elements[%d]: %d \n",y, InputArr8[y] );
     //for (uint32_t y=0;y<sizeInArr8;y++ ) printf(" Elements32[%d]: %d \n",y, InputArr32[y] );

     #endif          
}//endExpand



//Expand unsigned char Input into uint32_t to be processed. A*256+B .-->2_8
void inline  Expand_Pck2 (uint8_t *InputArr8, uint32_t *InputArr32 , uint32_t sizeInArr8){ //SizeInArr8  is the size of original input  8 bits array, no padding 

   __m128i load128;  
   __m512i load512,  Units, MuThs, Add;
   __m512i MASK_AVX512_SHIFTR_INT  =  _mm512_set_epi32(15,15,14,13,12,11,10,9, 8, 7, 6, 5,4,3,2,1);
   int step= 16-(STEP-1);
   __m512i Size = _mm512_set1_epi32(sizeInArr8);       //1. Fill  register Size  with the exact input size
   InputArr8[sizeInArr8]=0;InputArr8[sizeInArr8+1]=0;

   for (uint32_t i=0;i<sizeInArr8;i+=15 ){ //do{ (MRL-1)
	
	//printf ("here1");fflush(stdout);
        load128 = _mm_loadu_si128((__m128i *)(InputArr8+i));   	//1. Sequential Load,  8 bit int 
        load512 = _mm512_cvtepu8_epi32 (load128);   		//2. Expand to 32 bits
 
	//printf ("here2");fflush(stdout);
        //3. Pack the input
	MuThs = _mm512_slli_epi32 (load512, 7); //8, pero lo puse pror 7 para q sea 128 , pues el rango es 0..7                            // Generate thousands *256->=2_8 
	//printf ("here3");fflush(stdout);
        Units = _mm512_mask_permutexvar_epi32(MASKZERO_AVX512, 0x7FFF,MASK_AVX512_SHIFTR_INT,load512); //Calculates Units displacing original input
	//printf ("here4");fflush(stdout);
	Add = _mm512_add_epi32 (MuThs, Units);                  //3.- Sum both Values
	//printf ("here5");fflush(stdout);
        _mm512_store_epi32 (InputArr32+i,Add);   		//4. Store in new array.
	
	    #ifdef DEBUG_EXPAND_SIMD
		for (uint32_t j=i;j<i+MRL;j++ ) printf ("  InAr %u  ", InputArr8[j]);
		print_vector((void*)&load128,"     In128", "AVX512", "uint8_t", "DEC");printf("\n");
		print_vector((void*)&load512,"     In512", "AVX512", "uint32_t", "DEC");printf("\n");
		print_vector((void*)&MuThs,  "    In*256", "AVX512", "uint32_t", "DEC"); printf("\n");     		
		print_vector((void*)&Units,"       Units", "AVX512", "uint32_t", "DEC");printf("\n");
		print_vector((void*)&Add, "       Result", "AVX512", "uint32_t", "DEC");printf("\n");
		
		//print_vector((void*)&MuThs,"    MuThou2", "AVX512", "uint32_t", "DEC"); printf("\n");     		
		//print_vector((void*)&MuHs, "     MuHund", "AVX512", "uint32_t", "DEC");printf("\n");
		//print_vector((void*)&Mutns,"     MuTens", "AVX512", "uint32_t", "DEC");printf("\n");

	      	print_vector((void*)&Add,  "   Final Add", "AVX512", "uint32_t", "DEC");
		int c=getchar();
	    #endif
	//i+=(MRL-1);
	
   }//while (i<sizeInArr8);
   _mm512_store_epi32 (InputArr32+sizeInArr8, Size/*MASKZERO_AVX512, MASKFAKE_AVX512*/);            //Put 16 Higher al final de salida, 3 se considera el excedente *******Quizas no sea necesario.****


    #ifdef DEBUG_EXPAND
	
	//printf("\n");

        
   	//print_vector((void*)&Tens,    "Packed Values ", "AVX512", "uint32_t", "DEC");i
         int32_t aux8=0;
         for (uint32_t y=0;y<sizeInArr8;y++ ){ 
                      aux8 =  (InputArr8[y]*FACTORPACK)+(InputArr8[y+1]) ;
                      //printf(" Elements[%d]: %d, %d, %d, %d \n",y, InputArr8[y],InputArr8[y+1],InputArr8[y+2],InputArr8[y+3] );
                      if (aux8 !=InputArr32[y]) {  
                           printf(" ExpandPACK Error Output32[%d]: %d  Vs %d (8 bits) \n",y, InputArr32[y], aux8 ); fflush(stdout);
		  	   int c=getchar();	
		      }
         }
     //for (uint32_t y=0;y<sizeInArr8;y++ ) printf(" Elements[%d]: %d \n",y, InputArr8[y] );
     //for (uint32_t y=0;y<sizeInArr8;y++ ) printf(" Elements32[%d]: %d \n",y, InputArr32[y] );

     #endif          
}//endExpand_Pck2





//Version optimized for SACAS. For a general vision of any string checks the notebook. 
//Radixsort sorts starting from the char at the most right position to the left. An additional single character is incorporated  at every iteration. 
//It depends on selecting previously the correct characters to be ordered in every iteration. Sorts Suffixes not chars.Handles any size of input
void inline  PartialRadixSA_SIMD (uint32_t *InputArr, uint32_t sizeInArr, uint32_t *ExPsum, uint32_t *SortedArr , uint32_t *NewSortedArr, int64_t Inf, int64_t Sup, uint32_t *In, uint32_t *aux_VPI, uint16_t *aux_VLU ){
//El arreglo de datos inicia en CERO desde un punto de vista local. El arreglo de Sufijos inicia exactamente en la posicion de los elementos a ordenar y  hay paralelismo entre el arreglo de datos y el de sufijos

    __m512i VPI, VLU, VPI1, auxPS, auxAdd,Add1,Suff,data;//, outConf; 
    __mmask16 mymask, mymask2;
     uint32_t x,auxmask=0;

     uint32_t *auxNSArr=NewSortedArr+Inf;
     int64_t Fin; 
   	if ((Sup-Inf)>=MRL) Fin= Sup-MRL;
	else Fin= 0;
     //printf ("antes de SIMD\n");fflush(stdout);
     for (x=Inf; x<=Fin;x+=MRL){ 

        data = _mm512_load_epi32 ((void*) (InputArr+x)) ;

	VPI/*1*/ = _mm512_load_epi32 ( (aux_VPI+x));    //Both are parallel to data 
	mymask/*2*/ = aux_VLU[auxmask]; auxmask++;
		// Only if you find out that calculating is cheaper than storing/loading

        /*__m512i outConf = _mm512_conflict_epi32(data); 
        VPI = VPIgenerate (outConf);
        mymask2 = VLUgenerateMASK(outConf)^mask1 ;//VLU = VLUgenerate(outConf);*/

	/*printf ("Calculada %u Deseada %u\n",mymask2,mymask);
	print_vector((void*)&VPI,"CAL    VPI", "AVX512", "uint32_t", "DEC");
	print_vector((void*)&VPI1,"DES    VPI", "AVX512", "uint32_t", "DEC");
	int c=getchar();*/
        
        //mymask = _mm512_cmpeq_epu32_mask (VLU,MASKONE_AVX512); //2. Compress VLU
	//printf ("3 ");fflush(stdout);
        //3. Gather from Psum al the indexes in VPI  
        auxPS= _mm512_i32gather_epi32 (data, (void *) ExPsum, 4);
	//printf ("4 ");fflush(stdout);
        //4 Adds VPI and auxPrSum, these are the offsets
        auxAdd = _mm512_add_epi32 (auxPS, VPI);
	//printf ("5 ");fflush(stdout);
        Suff = _mm512_load_epi32 ((void*) (SortedArr+x)) ;               //Suffixes to sort
	//printf ("6 ");fflush(stdout);
        
	//print_vector((void*)&data,     "DATA       ", "AVX512", "uint32_t", "DEC");
	//print_vector((void*)&VPI,      "VPI       ", "AVX512", "uint32_t", "DEC");  
	//printf ("VLU Calc %u VLU DES %u \n", mymask,mymask2);
	//for (int m=0 ; m< 16; m++) printf (" ExPs [%i]=%u ", InputArr[m+x], ExPsum[InputArr[m+x]]);
	//print_vector((void*)&auxPS,     "auxPS       ", "AVX512", "uint32_t", "DEC");     int c= getchar();
 	_mm512_i32scatter_epi32 ((void *) auxNSArr/*NewSortedArr*/, auxAdd,Suff, 4) ;//Creates the partial order  of suffixes
	//printf ("7 ");fflush(stdout);
        Add1 = _mm512_add_epi32 (auxAdd, MASKONE_AVX512);                       //Adds the current Values of VPI and Corresponding prefix sum positions
	//printf ("8 ");fflush(stdout);
        _mm512_mask_i32scatter_epi32 ((void *)ExPsum, mymask,data , Add1, 4) ;  //Updates PrefixSum. //Mask is needed to avoid collisions cause data (indexing here) could have repeated elements, updating only the according to the last appeareance



  }//end of for x=0
	//printf ("antes de SEQ PR\n");fflush(stdout);

	for (uint32_t auxIt=x;auxIt<Sup;auxIt++){
		//printf ("soy OrdSeq SEQ, Colocando el sufijo %u, auxIt %d Inf %u Sup %d\n", SortedArr[auxIt],auxIt, Inf, Sup);
		//printf ("Posicion de escritura %u, caracter %u Inf %u Sup %d tamano Entrada %u, auxIt %u \n", ExPsum[ InputArr[auxIt] ], InputArr[auxIt], Inf, Sup, sizeInArr, auxIt);
		NewSortedArr[  ExPsum[ InputArr[auxIt] ] +Inf ]= SortedArr[auxIt];
		ExPsum[ InputArr[auxIt] ]++ ;//Update psum
	 		
	} //endfor auxit

	//printf ("Despues de SEQ PR");fflush(stdout);
	#ifdef DEBUG_PR //Partial Radix         
	  for (int i=Inf; i<Sup-1;i++){ 
	      //if (SortedArr[i-1]>SortedArr[i]) printf ("Error of sorting in SortedArr[%d]= %d\n ",i, SortedArr[i]) ; 
	      if (In[NewSortedArr[i]]>In[NewSortedArr[i+1]]){
	 printf (" Mal orden Inf %u Sup %u ParRadix pos[%u], suf=%u ValNewSor=%u, suf+1=%u, - ValNewSor[+1]=%u, \n ",Inf, Sup,i, NewSortedArr[i],In[NewSortedArr[i]], NewSortedArr[i+1], In[NewSortedArr[i+1]]) ; 
		//int c = getchar ();	
		}	      
	  }
	#endif
  #ifdef PRINT_PR //Partial Radix         
      for (int i=0; i<sizeInArr;++i){ 
   	 //if (SortedArr[i-1]>SortedArr[i]) printf ("Error of sorting in SortedArr[%d]= %d\n ",i, SortedArr[i]) ; 
   	printf (" NewSortedArr[%d]= %d \t ",i, NewSortedArr[i]) ; 
      }
   	
  #endif

}//end of PartialRadixSA

/**************************************************************SEQUENTIAL****************************************************************************/
void gen_Indexes(uint32_t *array, uint32_t size){
	uint32_t i;
	for (i=0;i<size;i++){
		array[i] = i;
		/*#ifdef DEBUG
		printf("%d\n", i);
		#endif*/
	}
}


uint32_t GroupBuilder (uint32_t *InputText, uint32_t size, uint32_t Iter, uint32_t *Indexes, uint32_t *OutInt, int64_t *frontera, uint8_t *HU, uint32_t *ID){
	   uint32_t subgrupos = 1;											;
	   int64_t It;//uint32_t It;
	   uint32_t aux=0;  	   /*aux= caracter valido previo*/
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
		//printf(" %u ", It);
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
			aux = (uint32_t) InputText [  ValidChar -1 ];
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
				aux = (uint32_t) InputText [  ValidChar -1 ];
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






/*****************************FOR DEBUGGING *****************************/
//Used for test to print the content of registers.
void print_vector(void *vec_ptr, std::string comment, std::string simdtype, std::string datatype, std::string format) {
  int32_t simd_size =0;
  if(simdtype == "SSE42") {
     simd_size = 128;
  } else if (simdtype == "AVX2") {
     simd_size = 256;
  } else if (simdtype == "AVX512") {
     simd_size = 512;
  } else {
      std::cout << "ERROR, " << simdtype << " is not recognized" << std::endl;
      exit(-1);
  }

  int32_t simd_segments =0;
  if(datatype == "uint8_t" || datatype == "int8_t") {
     simd_segments = simd_size/8;
  } else if (datatype == "uint16_t" || datatype == "int16_t") {
     simd_segments = simd_size/16;
  } else if (datatype == "uint32_t" || datatype == "int32_t") {
     simd_segments = simd_size/32;
  } else {
      std::cout << "ERROR, " << datatype << " is not recognized" << std::endl;
      exit(-1);
  }

  if(datatype == "uint8_t" ) {
     simd_segments = simd_size/8;
     uint8_t * res = (uint8_t*)vec_ptr;
     std::cout << comment << " : ";
     for(int32_t i=0; i< simd_segments; i++) {
        if (format == "HEX") {
           printf("[%d]=%0hhx ",i,res[i]);
        } else if(format == "DEC") {
           printf("[%d]=%0u ",i,res[i]);
        } else {
           printf("Error format does not exist\n");
           exit(1);
        }
    }
    std::cout << std::endl;
  } else if(datatype == "int8_t" ) {
     simd_segments = simd_size/8;
     int8_t * res = (int8_t*)vec_ptr;
     std::cout << comment << " : ";
     for(int32_t i=0; i< simd_segments; i++) {
        if (format == "HEX") {
           printf("[%d]=%0hhx ",i,res[i]);
        } else if(format == "DEC") {
           printf("[%d]=%0d ",i,res[i]);
        } else {
           printf("Error format does not exist\n");
           exit(1);
        }
    }
    std::cout << std::endl;
  } else if(datatype == "uint16_t" ) {
     simd_segments = simd_size/16;
     uint16_t * res = (uint16_t*)vec_ptr;
     std::cout << comment << " : ";
     for(int32_t i=0; i< simd_segments; i++) {
        if (format == "HEX") {
           printf("[%d]=%hx ",i,res[i]);
        } else if(format == "DEC") {
           printf("[%d]=%0u ",i,res[i]);
        } else {
           printf("Error format does not exist\n");
           exit(1);
        }
     }
     std::cout << std::endl;
  } else if(datatype == "int16_t" ) {
     simd_segments = simd_size/16;
     int16_t * res = (int16_t*)vec_ptr;
     std::cout << comment << " : ";
     for(int32_t i=0; i< simd_segments; i++) {
        if (format == "HEX") {
           printf("[%d]=%hx ",i,res[i]);
        } else if(format == "DEC") {
           printf("[%d]=%0d ",i,res[i]);
        } else {
           printf("Error format does not exist\n");
           exit(1);
        }
     }
     std::cout << std::endl;
  } else if(datatype == "uint32_t" ) {
     simd_segments = simd_size/32;
     uint32_t * res = (uint32_t *)vec_ptr;
     std::cout << comment << " : ";
     for(int32_t i=0; i< simd_segments; i++) {
        //std::cout << "[" << i << "]=" << res[i] << " ";
         if (format == "HEX") {
           printf("[%d]=%0x ",i,res[i]);
        } else if(format == "DEC") {
           printf("[%d]=%0u ",i,res[i]);
        } else {
           printf("Error format does not exist\n");
           exit(1);
        }
     }
     std::cout << std::endl;
  } else if(datatype == "int32_t" ) {
     simd_segments = simd_size/32;
     int32_t * res = (int32_t *)vec_ptr;
     std::cout << comment << " : ";
     for(int32_t i=0; i< simd_segments; i++) {
         if (format == "HEX") {
           printf("[%d]=%0x ",i,res[i]);
        } else if(format == "DEC") {
           printf("[%d]=%0d ",i,res[i]);
        } else {
           printf("Error format does not exist\n");
           exit(1);
        }
     }
     std::cout << std::endl;
  } else {
     std::cout << "ERROR, " << datatype << " is not a valid data type." << std::endl;
     exit(1);
  }
}


void Debug_MemC(uint32_t *Suffixes, uint32_t *NewSortedSuf, int64_t Inf, int64_t Sup){
	for (int i=Inf; i<Sup-1;++i){ 
	      if (NewSortedSuf[i]!= Suffixes[i]){ printf( "Error Memcopy pos: %u, Suf=%u vs NewSS=%u  \n",i, NewSortedSuf[i], Suffixes[i] ); int c = getchar();};
	}
} 
//

			
void DebugHistogram(uint32_t *Hist,uint32_t *NewSortedSuf, uint32_t *Input, int64_t Inf, int64_t Sup){
	int TestH[SIZEHIST];
        for (int32_t i=0; i<SIZEHIST;++i)   TestH[i]=0;   
       	for (int32_t i=Inf; i<Sup;++i) TestH[Input[NewSortedSuf[i]]] ++;
        for (int32_t i=0; i<SIZEHIST;++i)   
    	if ((TestH[i]!=Hist[i])&&(i<Sup))  printf ("After PartialRad ERROR DesiredHistog[%d]= %d vs Calculated[%d]= %d \n",i, Hist[i],i,TestH[i]) ; 

}

void SAtoFile(uint8_t *InputOriginal, uint32_t *Suffixes, uint32_t sizeInput ){
		char out_file[]="SA.dat";
		FILE *outfp = fopen(out_file, "w");
	  	if (outfp == NULL)    	{
	      		fprintf(stderr, "Error opening in file\n");
	      		exit(1);
	    	}
		for(uint32_t i=0;i<sizeInput;i++){
				//fprintf(outfp,"\t \tsufijo # %i pos: %i \n",i, Indexes[i]);
				fprintf(outfp,"%s\n", &InputOriginal[Suffixes[i]]);		
		}
		fclose(outfp);
	};


void ArraystoFile(uint32_t *InputOriginal, uint32_t *Suffixes, uint32_t sizeInput ){
		char out_file[]="Array.dat";
		FILE *outfp = fopen(out_file, "w");
	  	if (outfp == NULL)    	{
	      		fprintf(stderr, "Error opening in file\n");
	      		exit(1);
	    	}
		for(uint32_t i=0;i<sizeInput;i++){
				//fprintf(outfp,"\t \tsufijo # %i pos: %i \n",i, Indexes[i]);
			//	fprintf(outfp,"Suf %u  NSSuf %u iter %u \n", InputOriginal[i],Suffixes, i);		
		}
		fclose(outfp);
	};


void gen_data(uint8_t *data, uint32_t size){
	uint32_t i;
	uint8_t num;
	int seed = time(NULL);
	srand(seed);
	for (i=0;i<size;i++){
		num = (uint8_t)((rand()%5) + 97);//desde 65 a 90.. 
		#ifdef DEBUG_GEN 
		printf("IN--%d\n", num);
		#endif
		data[i] = num;
	}
	data[size] = '\0';
	#ifdef DEBUG_GEN 
	   FILE *outFile=fopen ("IntegerInput.txt", "w"); 
		for (i=0;i<size;i++) fprintf(outFile,"%d\n", data[i] );
   	   if (outFile) fclose(outFile);
	#endif

	//printf("%s\n", array);		
}





/****************************************************UNUSED**********************************************************/

void inline  GetHistogram (uint32_t *InputArr, uint32_t sizeInArr, uint32_t *Hist, uint32_t *VPI_Arr, uint32_t *VLU_Arr){

   uint32_t *auxInput;  
   __m512i data,outConf, VPI, VLU, VPI1, currentData, _Add;  
   __mmask16 mymask, mymask_, mymask2;

   for (uint32_t x=0; x<sizeInArr;x+=16){ 

        auxInput=(InputArr+x);                
        data = _mm512_load_epi32 ((void*) auxInput) ;

        outConf = _mm512_conflict_epi32(data);        
        VPI= VPIgenerate(outConf);
        mymask = VLUgenerateMASK(outConf)^mask1 ; //VLU = VLUgenerate(outConf);
        VPI1 = _mm512_add_epi32(MASKONE_AVX512,VPI);   


        currentData = _mm512_mask_i32gather_epi32 (MASKZERO_AVX512, mymask,data , (void const *)Hist,4); //bring indexes from GLOBALHist in memory
        //si el SAC es únicamente para genómica se pueden eliminar todas las cubetas y hacer el cálculo del histograma directamente sobre Registros
        _Add = _mm512_add_epi32 (currentData, VPI1);                         //Sum The local Histogram (just calculated) with the previously existing 
        _mm512_mask_i32scatter_epi32 ((void *)Hist, mymask,data , _Add, 4) ; //Update Global Hist, needs the mask for the one that has been added

	#ifdef PRINT_H         
	   printf ("********* Histogram's Hiteration***** %u, SizeInArr %d \n ", x, sizeInArr);
           printf("mymask Hexa = %x \n",mymask);   
           print_vector((void*)&data,    "Data_Index", "AVX512", "uint32_t", "DEC");
           //print_vector((void*)&outConf, "OutCnf    ", "AVX512", "uint32_t", "DEC");
           print_vector((void*)&VLU,     "VLU       ", "AVX512", "uint32_t", "DEC");           
           print_vector((void*)&VPI,     "VPI       ", "AVX512", "uint32_t", "DEC");
           //print_vector((void*)&VPI1,    "VPI+1     ", "AVX512", "uint32_t", "DEC");   
           print_vector((void*)&currentData, "DatFromMem", "AVX512", "uint32_t", "DEC");
           print_vector((void*)&_Add,        "   SumHist", "AVX512", "uint32_t", "DEC");
           printf ("\n");
        #endif
   }//end for x=0
  
   #ifdef DEBUG_H         
      //for (int32_t i=0; i<sizeInArr;++i)  if( InputArr[i]>0)printf ("Input[%d]= %d  e Hist = %d \n",i, InputArr[i], Hist[InputArr[i]]) ; /*int c=getchar();*/
      int TestH[SIZEHIST];
      for (int32_t i=0; i<256;++i)   TestH[i]=0;   
      for (int32_t i=0; i<sizeInArr;++i)  TestH[ InputArr[i]  ]++;
      for (int32_t i=0; i<256;++i)   
 	    if (TestH[i]!=Hist[i])  { printf ("CalculatedH[%d]= %d ERROR vs DesiredHist[%d]= %d \n",i, Hist[i],i,TestH[i]) ; }
   #endif

}//end of GetHistogram






void inline  Psum (uint32_t *Hist, uint32_t *ExPsum){          //GLOBAL PREFIXSUM: Calculates ExPsum over the input Array, multiple of 16
         uint32_t *auxH, *aux1;
         __m512i Add, dat, LocExPSum ; 
         __m512i auxAdd = MASKZERO_AVX512;                       //Initialize the aux register that helps in calculating the global Psum

         for (int x=0; x<256;x+=MRL){   
                auxH=(Hist+x);

	       dat=_mm512_load_epi32 ((void *) auxH);   	 

 	       LocExPSum= _mm512_PrefixSum_epi32(dat);        //Calculate localReg ExPsum
 
               Add = _mm512_add_epi32 (LocExPSum,auxAdd);	 //Add with the previous iteration Psum and this is the real general Xpsum
 		 
               _mm512_store_epi32 ((void*) (ExPsum+x),Add);

               auxAdd = _mm512_add_epi32(Add, dat); 		//Updates the auxiliarReg  for next iteration
               aux1 = (uint32_t*)&auxAdd ;  

               //La otra opcion si se quiere mantener todo en registros es :  haciendo un shuffle de la ultima posicion a todas las de AuxAdd? How?
               auxAdd = _mm512_set1_epi32(aux1[15]); //o accederlos aca directamente ?  

              #ifdef PRINT_PS
                   //cout << "ExPrefixSum Iteration " << x/16 << "de 16\n";
                   //print_vector((void*)&dat, "InputDat", "AVX512", "uint32_t", "DEC");
                   //print_vector((void*)&LocExPSum, "LocExPSum", "AVX512", "uint32_t", "DEC");    
                   //print_vector((void*)&Add, "FinalPSum", "AVX512", "uint32_t", "DEC");           
                   __m512i Test= _mm512_shuffle_epi8 (auxAdd, ShufflePsum);
  		   cout << "Acarreo del prefix Sum Ex " <<aux1[15] << " \n "; 
                   print_vector((void*)&auxAdd, "auxAdd", "AVX512", "uint32_t", "DEC");          
                   print_vector((void*)&Test,   "TestSh", "AVX512", "uint32_t", "DEC");         
 	           int c=getchar(); 
               #endif 
         }//end for x=0
       
         #ifdef DEBUG_PS
             uint32_t auXPsum[SIZEHIST];
             auXPsum[0]=Hist[0];  
             if (0!=ExPsum[0] ) printf ("Error ExPsumCalc[%d]= %d ERROR vs ExPsumDes[%d]= %u \n",0, ExPsum[0],0,auXPsum[0]) ;
             for (int x=1; x<256;++x){  
                auXPsum[x]=auXPsum[x-1]+Hist[x];
                if (auXPsum[x]!=ExPsum[x] ) {printf ("Error ExPsumCalc[%d]= %d ERROR vs ExPsumDes[%d]= %u \n",x, ExPsum[x],x,auXPsum[x]);int c=getchar(); };
             }	   
            
         #endif 

} //end of ExcPsum  

void swapPointers(uint32_t **Suffixes, uint32_t **NewSortedSuf){
	uint32_t *temp=*Suffixes;	*Suffixes=*NewSortedSuf;  *NewSortedSuf=temp;	
}


