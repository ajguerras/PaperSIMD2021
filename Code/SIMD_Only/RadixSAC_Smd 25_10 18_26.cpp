//source /opt/intel/bin/compilervars.sh intel64
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
//#include <time.h>
#include <omp.h>


//Constants
#define ADJUST 1
#define MRL 16  //Maximu register length
#define BASE_BITS 8
#define MASK (BASE-1)
#define ENDCHAR 0
#define STEP 2
#define CARDINALIDAD 32//20000 //24000//32000//
#define CARD_TLP_ONLY
//#define SIZEHIST 66018 //Para packing de a dos elementos
//Importantes para modeificar con el packing
#define SIZEHIST (256+MRL) //Tamaño del histograma Sin packing (16400)
#define PASOITER 1
#define HIGHESTCHAR (256+MRL-1) //OJO ESTE UNO LO COLOQUE EL 23-10 por problemas en el Hist.


//Macros
#define BASE (1 << BASE_BITS)
#define DIGITS(v, shift) (((v) >> shift) & MASK)
#define MIN(X, Y) (((X) < (Y)) ? (X) : (Y))

//Activate Insersort
#define INSERTSORT
#define RANGE_ISORT MRL+MRL+1

//Debug constants
//#define archivos
//#define PRINT_PR_SIMD
//#define DEBUG
//#define DEBUG_GEN
#define DEBUG_IS
#define DEBUG_EXPAND
#define DEBUG_H
#define DEBUG_PS
#define DEBUG_SVC
#define DEBUG_PR
//#define DEBUG_H_SIMD
//#define DEBUG_MEMC
//#define DEBUG_SAC //Para generar la salida completa del archivo.
//#define DEBUG_per_ITER
//#define PRINT_PR

#define TESTS_MT //For automatic tests

#ifdef PACK_2
	#define SIZEHIST (16401) //
	#define PASOITER 2
	#define HIGHESTCHAR 16400 //
#endif


/*******************************   Constant AVX_512 Registers    **********************************/
   __m512i MASKZERO_AVX512  = _mm512_set_epi32(0x0, 0x0, 0x0,0x0,0x0,0x0,0x0,0x0,0x0, 0x0, 0x0,0x0,0x0,0x0,0x0,0x0);
   __m512i MASKONE_AVX512   = _mm512_set_epi32(0x1, 0x1, 0x1,0x1,0x1,0x1,0x1,0x1,0x1, 0x1, 0x1,0x1,0x1,0x1,0x1,0x1);
   __m512i MASK_AVX512_1to16 = _mm512_set_epi32(16,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1);          //Used  for Initializing Suffixes and SVC
   __m512i MASK_AVX512_16    = _mm512_set_epi32(16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16); //Used  for Initializing Suffixes and SVC
   __m512i MASK_AVX512_0to15 = _mm512_set_epi32(15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0);   
   __m512i MASKFAKE_AVX512_PACK   = _mm512_set_epi32(0x101C0, 0x101C0, 0x101C0,0x101C0,0x101C0,0x101C0,0x101C0,0x101C0,0x101C0, 0x101C0, 0x101C0,0x101C0,0x101C0,0x101C0,0x101C0,0x101C0); 
   __m512i MASKFAKE_AVX512   = _mm512_set_epi32(0xFF, 0xFF, 0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF, 0xFF, 0xFF,0xFF,0xFF,0xFF,0xFF,0xFF); 

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
void Seq_lsd_radix_sort_(uint32_t n, uint8_t *data, uint32_t *Indexes, int inf, int sup);
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
	
	sizeInput=10167;//32543;//50428182;//1892;//1016732543;//;//101673;//2543;//704281892;//1083792832;//2;//33120193;//222193//333120193;//
	uint8_t *InputOriginal; 
	uint32_t *Suffixes, NextSize; 

	/****Automated TEsts Only***********/   
	#ifdef TESTS_MT
		ofstream myfile;
		myfile.open ("SAC OPT MIMD ONLY Experiments Oct_18.txt",ios::app);
		myfile << " --------------------New Round of experiments OPT MIMD ONLY----------------------\n" << endl;
		myfile.close();	
		uint32_t NthreadsArr[5]={101691,1016732,10167325,101673254,1016732543};//11,1
		int len = sizeof(NthreadsArr) / sizeof(uint8_t);


	for (int i=0 ; i<len; i++){ 	//for (int i=len-1 ; i>=0; i--){
		sizeInput=NthreadsArr[i];///printf ("tamaño %u \n", sizeInput); fflush(stdout);
	#endif


	
        int succP= posix_memalign ((void **) &InputOriginal, 64, (sizeInput+16)*sizeof(uint8_t));
	if (succP!=0){         printf ("Error reserving memory for InputOriginal");            exit(0);    }  

	
    	succP= posix_memalign ((void **) &Suffixes, 64, (sizeInput+16)*sizeof(uint32_t)); 
        if (succP!=0){         printf ("Error reserving memory for Suffixes");            exit(0);    } 

	gen_data(InputOriginal, sizeInput);


	auto startLocal = std::chrono::high_resolution_clock::now();   
	Radix_SAC_512 (sizeInput,  InputOriginal, Suffixes );
	auto stopLocal = std::chrono::high_resolution_clock::now();   
	
	#ifdef TESTS_MT
	
		cout << (uint32_t )sizeInput << " Sufijos Ordenados,  Milliseconds in MIMD RadixSAC: "<< duration_cast<milliseconds>(stopLocal-startLocal).count() << ".\n" << endl;
		myfile.open ("SAC OSIMD  Experiments Oct_25.txt",ios::app);
		myfile << (uint32_t )sizeInput<< " Sufijos Ordenados,  Milliseconds in MIMD RadixSAC:"<< duration_cast<milliseconds>(stopLocal-startLocal).count() << ".\n" << endl;
		myfile.close();	
        //cout << (uint32_t) NextSize<<" Hilos "<< duration_cast<milliseconds>(stopLocal-startLocal).count() << " Chronos milliseconds in MIMDRadix (cout).\n" << endl;

	#ifdef DEBUG_SAC
		SAtoFile(InputOriginal, Suffixes, sizeInput );
		int c=getchar();
	#endif
	}
	#endif
	
	/************************************End of automated tests*****************************************/


	#ifdef DEBUG  
		int a=0;  char SizeInString[36];
		sprintf(SizeInString, "ErrorLogSACRadix512_%d.txt", sizeInput);
	
	 	FILE *outFile=fopen (SizeInString, "w"); 
	   	fprintf (outFile, "Starting Radixsort... \n ");fflush(stdout);
		//fprintf (outFile, "\n %lld Milliseconds in RADIXSIMD.\n",  duration_cast<milliseconds>(stopLocal-startLocal ).count()) ;*/
		
	     	for (int32_t i=0; i<sizeInput;++i){ 
			unsigned char *menor, *mayor;
		 	fprintf (outFile,"Suf# %i , FinalPos %d = %s\n",Suffixes[i],i, InputOriginal+Suffixes[i]) ; //If  you want to see all the suffixes on screen
		 	printf ("Suf# %i, FinalPos %d =    %s\n",Suffixes[i],i, InputOriginal+Suffixes[i]) ; //If  you want to see all the suffixes on scree
			menor=(unsigned char*)InputOriginal+Suffixes[i] ;  mayor=(unsigned char*)InputOriginal+Suffixes[i+1]; 
			a =strcmp((char *)menor, (char *)mayor);      
		 	if  ((i<(sizeInput-1))&&((*menor>*mayor)||(a>0))) {
			    fprintf(outFile,"\n ****ERROR GRAVE EN EL ORDENAMIENTO FINAL POS %i  \niMenor: %s \nMayor : %s\n ",i, menor , mayor); fflush(stdout); 
				
			    printf("\n ****ERROR GRAVE EN EL ORDENAMIENTO FINAL POS %i  menor: %s\n mayor : %s\n ",i, menor , mayor); fflush(stdout);   
			   int c = getchar ();               	
			}
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

	Expand_SIMD (InputOriginal, InputExp,  sizeInput); //CUIDADO , si SE HACE ALGUN PACK SE DEBE REDIMENSIONAR LA ENTRADA
	//Expand_Pck2 (InputOriginal, InputExp , sizeInput);

	Frontera[0] = -1;
	uint32_t auxSwap,x, subgrupos=1;
	uint32_t currentIter=0;// Inf=Frontera[x]+10, Sup=sizeInput (Frontera[x+1];
	uint8_t band=1;
	int64_t Inferior, Cardinalidad, aux = (int64_t)sizeInput-1;//, groupCounter=0, groupHugeCounter=0;

	memset(Hist,0,sizeH); 
	/*SelectValidCharSIMD (InputExp, sizeInput, 0, Suffixes, OutChar, 0, sizeInput);
	for (uint32_t It=0;It<sizeInput;++It){  //Para cada uno d elos caracterres validos    
	     if (OutChar[It]!=InputExp[It]) {
		printf("\nSVC  Iter %u OUTCHAR %u  Input=  %u    \n", It, OutChar[It], InputExp[It]  ); int c = getchar ();
   	     }	 	
	   
	}printf ("SVC done");fflush(stdout);*/


	GetHistogramSIMD (/*OutChar*/ InputExp, sizeInput, Hist, 0,  sizeInput,aux_VPI,aux_VLU);
	//printf ("HIST done");fflush(stdout);
	ExcPsum (Hist, ExPsum);	//printf("\t ExPSUM ");fflush(stdout);	
	//printf ("EPS done");fflush(stdout);

	//int c=getchar();
	//ArraystoFile(Suffixes, NewSortedSuf,  sizeInput); c=getchar();
	PartialRadixSA_SIMD (/*OutChar*/InputExp, sizeInput, ExPsum, Suffixes, NewSortedSuf, 0,  sizeInput,InputExp,aux_VPI,aux_VLU);
	//printf ("PARTIAL DONE");
	//ArraystoFile(Suffixes, NewSortedSuf,  sizeInput);/*int*/ c=getchar();
	//swapPointers(&Suffixes, &NewSortedSuf); //_mm512_store_epi32 (Suffixes+sizeInput,MASKZERO_AVX512); 
	//ArraystoFile(Suffixes, NewSortedSuf,   sizeInput); c=getchar();
	//SAtoFile(InputOriginal, NewSortedSuf/*Suffixes*/, sizeInput );c=getchar();
	

	memcpy(Suffixes, NewSortedSuf, (sizeInput<<2) );
	//SAtoFile(InputOriginal, Suffixes, sizeInput );
	//ArraystoFile(Suffixes, NewSortedSuf,  sizeInput); int c=getchar();
	//memcpy(Suffixes+(sizeInput-MRL), NewSortedSuf+(sizeInput-MRL), (32<<2) );
	currentIter+=PASOITER;	
	subgrupos = GroupBuilder (InputExp, sizeInput, currentIter, Suffixes, OutChar, Frontera, HU, ID);

	band=1; 
	do{//Aqui comienza el ciclo que produce el orden
		
		Frontera [subgrupos] = aux ; /*OJO cuidar*/
		band=1; //printf ("START"); fflush(stdout);

	
		for(x=0;x<subgrupos;x++){
			Cardinalidad= (int64_t) Frontera[x+1]-Frontera[x];
			if ( Cardinalidad>1 /*((MRL+1)*Nthreads)*/ ){ //Intervenir  y meter el Insersort para menos de 130 elementos.

				//printf ("ENTERing \n");fflush(stdout);
				memset(Hist,0,sizeH);  
				Inferior=Frontera[x]+1;

				if (Cardinalidad> 1/*(CARDINALIDAD)*/){

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
						if (InputExp[Suffixes[Inferior]]> InputExp[Suffixes[Frontera[x+1]]]){
							auxSwap=Suffixes[Inferior]; Suffixes[Inferior] = Suffixes[Frontera[x+1]]; Suffixes[Frontera[x+1]]=auxSwap;//Reordene los indices
							HU[Suffixes[Frontera[x+1]]]=1;HU[Suffixes[Inferior]]=1; 		//Marque HU	
						}
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
		subgrupos = GroupBuilder (InputExp, sizeInput, currentIter, Suffixes, OutChar, Frontera, HU, ID);//Deberia ser OutInt la entrada
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

       // _mm512_store_epi32 ((OutInt+(Sup)),MASKFAKE_AVX512); //1 more time, for the fakes values, igher so it does not affect the prefix sum 
 


	 #ifdef DEBUG_SVC
	   for (uint32_t It=Inf;It<Sup;++It){  //Para cada uno d elos caracterres validos    

		uint32_t aux, ValidChar = Indexes[It] + currentIter;
		if(!(ValidChar >= Size)){       //PAdding 
		    aux = InputText [  ValidChar ] ;  
		}else{
		    aux = 255;//HIGHESTCHAR;		//reevaluar, Caracter Invalido. Debe ser cero o debe ser un numero muy grande para que no afecte el prefix sum
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

   for ( x=Inf; x<Fin;x+=MRL){ 

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
           print_vector((void*)&VLU,     "VLU       ", "AVX512", "uint32_t", "DEC");           
           print_vector((void*)&VPI,     "VPI       ", "AVX512", "uint32_t", "DEC");
           //print_vector((void*)&auxPOP,  "auxPOP    ", "AVX512", "uint32_t", "DEC");
           //print_vector((void*)&VPI1,    "VPI+1     ", "AVX512", "uint32_t", "DEC");   
           print_vector((void*)&currentData, "DatFromMem", "AVX512", "uint32_t", "DEC");
           print_vector((void*)&_Add,        "   SumHist", "AVX512", "uint32_t", "DEC");
           int c=getchar();
        #endif
   }//end for x=0
   for (uint32_t auxIt=x;auxIt<Sup;auxIt++){ 
		//printf ("soy HIST SEQ, auxIt %d Sup %d\n", auxIt, Sup);
		Hist[InputArr[auxIt]]++;
	}

   #ifdef DEBUG_H         
      //for (int32_t i=0; i<sizeInArr;++i)  if( InputArr[i]>0)printf ("Input[%d]= %d  e Hist = %d \n",i, InputArr[i], Hist[InputArr[i]]) ; /*int c=getchar();*/
      int TestH[SIZEHIST];
      for (int32_t i=0; i<SIZEHIST;++i)   TestH[i]=0;   
      for (int32_t i=Inf; i<Sup;++i)  TestH[ InputArr[i]  ]++;
      for (int32_t i=0; i<255;++i)   
 	    if (TestH[i]!=Hist[i])  { printf ("CalculatedH[%d]= %d ERROR vs DesiredHist[%d]= %d \n",i, Hist[i],i,TestH[i]) ; int c=getchar();}
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
                      aux8 =  (InputArr8[y]*256)+(InputArr8[y+1]) ;
                      //printf(" Elements[%d]: %d, %d, %d, %d \n",y, InputArr8[y],InputArr8[y+1],InputArr8[y+2],InputArr8[y+3] );
                      if (InputArr8[y] !=InputArr32[y]) {  
                           printf(" Expand Error Output32[%d]: %d  Vs %d (8 bits) \n",y, InputArr32[y], aux8 ); fflush(stdout);
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

   for (uint32_t i=0;i<sizeInArr8;i+=MRL ){ 

        load128 = _mm_loadu_si128((__m128i *)(InputArr8+i));   	//1. Sequential Load,  8 bit int 
        load512 = _mm512_cvtepu8_epi32 (load128);   		//2. Expand to 32 bits
 
        //3. Pack the input
	MuThs = _mm512_slli_epi32 (load512, 7); //8, pero lo puse pror 7 para q sea 128                                                       // Generate thousands *256->=2_8 
        Units = _mm512_mask_permutexvar_epi32(MASKZERO_AVX512, 0x7FFF,MASK_AVX512_SHIFTR_INT,load512); //Calculates Units displacing original input

	Add = _mm512_add_epi32 (MuThs, Units);                  //3.- Sum both Values

        _mm512_store_epi32 (InputArr32+i,Add);   		//4. Store in new array.
   }
   _mm512_store_epi32 (InputArr32+sizeInArr8, Size/*MASKZERO_AVX512, MASKFAKE_AVX512*/);            //Put 16 Higher al final de salida, 3 se considera el excedente *******Quizas no sea necesario.****


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
                      aux8 =  (InputArr8[y]*256)+(InputArr8[y+1]) ;
                      //printf(" Elements[%d]: %d, %d, %d, %d \n",y, InputArr8[y],InputArr8[y+1],InputArr8[y+2],InputArr8[y+3] );
                      if (aux8 !=InputArr32[y]) {  
                           printf(" Expand Error Output32[%d]: %d  Vs %d (8 bits) \n",y, InputArr32[y], aux8 ); fflush(stdout);
		  	   //int c=getchar();	
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
     //printf ("antes de SIMD");fflush(stdout);
     for (x=Inf; x<=Fin;x+=MRL){ 

        data = _mm512_load_epi32 ((void*) (InputArr+x)) ;

	VPI/*1*/ = _mm512_load_epi32 ( (aux_VPI+x));    //Both are parallel to data 
	mymask/*2*/ = aux_VLU[auxmask]; auxmask++;
		// Only if you find out that calculating is cheaper than storing/loading

        /*__m512i outConf = _mm512_conflict_epi32(data); 
        VPI = VPIgenerate (outConf);
        mymask = VLUgenerateMASK(outConf)^mask1 ;//VLU = VLUgenerate(outConf);

	printf ("Calculada %u Deseada %u\n",mymask2,mymask);
	print_vector((void*)&VPI,"CAL    VPI", "AVX512", "uint32_t", "DEC");
	print_vector((void*)&VPI1,"DES    VPI", "AVX512", "uint32_t", "DEC");
	int c=getchar();*/
        
        //mymask = _mm512_cmpeq_epu32_mask (VLU,MASKONE_AVX512); //2. Compress VLU

        //3. Gather from Psum al the indexes in VPI  
        auxPS= _mm512_i32gather_epi32 (data, (void *) ExPsum, 4);

        //4 Adds VPI and auxPrSum, these are the offsets
        auxAdd = _mm512_add_epi32 (auxPS, VPI);

        Suff = _mm512_load_epi32 ((void*) (SortedArr+x)) ;               //Suffixes to sort
 	_mm512_i32scatter_epi32 ((void *) auxNSArr/*NewSortedArr*/, auxAdd,Suff, 4) ;//Creates the partial order  of suffixes

        Add1 = _mm512_add_epi32 (auxAdd, MASKONE_AVX512);                       //Adds the current Values of VPI and Corresponding prefix sum positions
        _mm512_mask_i32scatter_epi32 ((void *)ExPsum, mymask,data , Add1, 4) ;  //Updates PrefixSum. //Mask is needed to avoid collisions cause data (indexing here) could have repeated elements, updating only the according to the last appeareance



  }//end of for x=0
	//printf ("antes de SEQ PR");fflush(stdout);

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
    	/*if (TestH[i]!=Hist[i])*/  printf ("After PartialRad ERROR DesiredHistog[%d]= %d vs Calculated[%d]= %d \n",i, Hist[i],i,TestH[i]) ; 

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
				fprintf(outfp,"Suf %u  NSSuf %u iter %u \n", InputOriginal[i],Suffixes, i);		
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


//Version optimized for SACAS. For a general vision of any string checks the notebook for a LSC approach. 
//Radixsort sorts starting from the char at the most Left position. and goes to the right. Desgined to work with groups betwwen Inf (inclusive) and Sup (exclusive)
//It depends on selecting previously the correct characters to be ordered in every iteration. Sorts Suffixes not chars.Handles any size of input
void inline PartialRadixSAC_MIMD (uint32_t *InputArr, uint32_t sizeInArr, uint32_t *ExPsum, uint32_t *SortedArr , uint32_t *NewSortedArr, int64_t Inf, int64_t Sup, uint32_t Nthreads, uint32_t *In){
//El arreglo d datos inicia en CERO desde un punto de vista local. El arreglo d Sufijos inicia exactamente en la posicion d los elementos a ordenar. Hy paralelismo entre el arreglo d datos y d sufijos
   uint32_t ChunkSize=(uint32_t) ((Sup-Inf)/Nthreads)+1;

	//printf("\nSTART Inf %u Sup %u Chun %u ------------------------------------\n", Inf, Sup, ChunkSize);
	


   #pragma omp parallel  num_threads(Nthreads) shared(ChunkSize) //firstprivate(Res,Displacement)
   { 
	int tid = omp_get_thread_num(); 
  	uint32_t x,Inicio= (tid*ChunkSize)+Inf;
	uint32_t Fin = MIN(Sup-MRL , (Inicio + ChunkSize-MRL));  // Ver abajo el procesamiento del bloque remanente
	if (ChunkSize<MRL) Fin =0;			//OPT Esta linea se puede quitar siempre que el grupo sea mayor a 17*NT= 1500 en nuestro caso
							//	De lo conrario procesa los elementos repetidamente cuando los Hilos tienen menos de 16 elementos.
	__m512i VPI1, VPI, VLU, auxPS, auxAdd,Add1,Suff,data, outConf;
    	__mmask16 mymask;
	//***__m512i Displacement = _mm512_set1_epi32(Inicio); //Solo hace falta para ordenamiento paralelo que ademas es por grupos
	printf ("PARALLEL Soy %d, Inicio =%u, Fin = %u\n",tid,Inicio,Fin);fflush(stdout);

	uint32_t *auxNSArr=(NewSortedArr+Inf);

	for (x=Inicio; x<Fin;x+=MRL){ //Debe restarele 16 ...  //*********ojo toca sUMAR EL inicio a la posicion de orden., Pilas, a QUIEN ?

		data = _mm512_load_epi32 ((void*) (InputArr+x)) ;
				//VPI = _mm512_load_epi32 ( (VPI_Arr+x));    //Both are parallel to data 
				//VLU = _mm512_load_epi32 ( (VLU_Arr+x));
				// Only if you find out that calculating is cheaper than storing/loading

		outConf = _mm512_conflict_epi32(data); 
		VPI = VPIgenerate (outConf);
		mymask = VLUgenerateMASK(outConf)^mask1;

		VPI1 = _mm512_add_epi32 (MASKONE_AVX512, VPI); // B
		#pragma omp critical (PRad)
		{
			auxPS= _mm512_i32gather_epi32 (data, (void *) ExPsum, 4);         //3. Gather from Psum al the indexes in VPI  . 
											  //Permanece intacto los limites xq la estructura EXpsum es local a este grupo.
			///*A*/auxAdd = _mm512_add_epi32 (auxPS, VPI);        			  //4 Adds VPI and auxPSum, these are the offsets
			//auxAdd = _mm512_add_epi32 (auxAdd, Displacement);        	//Solo hace falta para ordenamiento paralelo que ademas es por grupos
			///*A*/Add1 = _mm512_add_epi32 (auxAdd, MASKONE_AVX512);                       //Adds the current Values of VPI and Corresponding prefix sum positions

			Add1=_mm512_add_epi32 (VPI1, auxPS);  //B 
			
			_mm512_mask_i32scatter_epi32 ((void *)ExPsum, mymask,data , Add1/*auxAdd*/, 4) ;  //Updates ExPSum. VLU Mask is used to avoid collisions, according to the last appeareance nly
		}//end pragma critical

		auxAdd=_mm512_sub_epi32(Add1,MASKONE_AVX512);//B
		Suff = _mm512_load_epi32 ((void*) (SortedArr+x)) ;                //5. Read the suffixes /**/to sort

	 	//_mm512_i32scatter_epi32 ((void *) NewSortedArr, auxAdd,Suff, 4) ; //6. Creates the partial order  of suffixes, en las posiciones auxAdd
		_mm512_i32scatter_epi32 ((void *) auxNSArr, auxAdd,Suff, 4); //*************For group order starting in inF


			#ifdef PRINT_PR_SIMD          
				   printf ("Soy %d, Inicio =%lu, Fin = %lu\n",tid,Inicio,Fin);fflush(stdout);
				   printf ("*********Internal Partial Sorting Iteration***** soy %d x=%u\n ", tid, x);     
				   print_vector((void*)&data,     "      Data", "AVX512", "uint32_t", "DEC");
				   //print_vector((void*)&outConf, "outCnf     ", "AVX512", "uint32_t", "DEC");
				   //print_vector((void*)&VLU,     "        VLU", "AVX512", "uint32_t", "DEC");            
				   print_vector((void*)&VPI,     "        VPI", "AVX512", "uint32_t", "DEC");
		  		   VLU = VLUgenerate(outConf);  
				   __mmask16 maskdesired = _mm512_cmpeq_epu32_mask (VLU,MASKONE_AVX512);  //colocar operación de VLU optimizada               //Calculates the mask compressing VLU
				   printf("mymask Hexa Calculated = %x vs desired %x\n",mymask, maskdesired );
				   print_vector((void*)&auxPS,   "DatFromPsum", "AVX512", "uint32_t", "DEC");          
				   print_vector((void*)&auxAdd,  "Offsets    ", "AVX512", "uint32_t", "DEC");          
				   print_vector((void*)&Add1,    "UpdatedPsum", "AVX512", "uint32_t", "DEC");
				   print_vector((void*)&Suff,    "Sufijos    ", "AVX512", "uint32_t", "DEC");          
				   //print_vector((void*)&data,   "TargtIndex ", "AVX512", "uint32_t", "DEC");
				   printf ("\n");          int c = getchar();
				   //for (int i=0; i<sizeInArr;++i)           	printf ("SortedArr[%d]= %d,\n ",i, SortedArr[i]) ;
				#endif


  	}//end of for x

/*
	uint32_t Fin = MIN(Sup, (Inicio + ChunkSize));  // Ver abajo el procesamiento del bloque remanente
	x=Inicio;
	if (ChunkSize>MRL) {			//OPT Esta linea se puede quitar siempre que el grupo sea mayor a 17*NT= 1500 en nuestro caso
		Fin=Fin-MRL;						//	De lo conrario procesa los elementos repetidamente cuando los Hilos tienen menos de 16 elementos.

		
			#pragma omp critical //(ParRad)
     
	#pragma omp critical //(ParRad)  //No estoy seguro de si este es necesario o solo el de adentro
} */



       //Procesamiento del Remanente
       //Opcion 1: Restituir la copia del segmento del final , uno pudiera pensar que no importa q se dañe SVC al final (porque se genera iterativametne), entnces se ahorrarian muchos pasos  es 		cierto, pero es preferible cuidar la integridad de SVC, a menos que HU estuviera muy bien oranizado.
       //Hacer una copia del segmento del final , this might not be enough due to the scatter process, El secreto esta en q SELECTVALID CHAR debe pasar FAKE VALUES AL FINAL. El problema esta en     
       //q  SVC Construye una sola estructura GLOBAL porque esto es lo que necesitara el group builder

       //Opcion 2: La otra opcion es hacer el ordenamiento del remanente iterativamente, estos valores estan el EXPsum by the way, tocaria sumarle uno por cada aparicion para obtener el offset, y 	
       //ademas actualiar el Psum por sia

       
	Fin= MIN(Sup, (Inicio + ChunkSize)); 
	//printf ("SEQ       Soy %d, Inicio =%u, Fin = %u, It= %u, Sup = %u\n",tid,Inicio,Fin, x, Sup);fflush(stdout);
	#pragma omp critical (PRad)  //No estoy seguro de si este es necesario o solo el de adentro
	{
		for (uint32_t auxIt=x;auxIt<Fin;auxIt++){

			NewSortedArr[  ExPsum[ InputArr[auxIt] ] +Inf ]= SortedArr[auxIt];
			//printf ("CRITIC soy %i El elemento valor= %u va a la posicion d orden [%i] SufijOld %i Sufijo Nvo %u\n", tid, InputArr[auxIt],ExPsum[ InputArr[auxIt] ] +Inf, SortedArr[auxIt],NewSortedArr[  ExPsum[ InputArr[auxIt] ] +Inf ]);fflush(stdout);	
			//pragma omp critical
			ExPsum[ InputArr[auxIt] ]++ ;//Update psum
	 		

		} //endfor auxit
        }//EndCritical 
	//for (uint32_t auxIt=Inicio;auxIt<Fin;auxIt++) {printf ("Proc %i El elemento de la posicion [%u] es %i, sufijo %u \n", tid, auxIt, InputArr[NewSortedArr[auxIt]],NewSortedArr[auxIt]);fflush(stdout);} 

	/*#pragma omp critical
		insertionSort(InputArr, SortedArr, Inicio,  Fin);*/
   

       //Opcion 3: La otra es hacer el ultimo pedacito aparte pero con SIMD. cuidando no meter data dañina
       /*uint32_t *auxData,succP; 
       succP= posix_memalign ((void **) &auxData, 64, 33*sizeof(uint32_t));  
	
       //data = _mm512_load_epi32 ((void*) (InputArr+x)) ;	//Read from valid data, 
       //_mm512_store_epi32 (auxData, data);		        //Storevalid data
	
       memcpy (auxData, InputArr+x, 32);	//(**********Cuidar dimensiones, MRL*sizeof(uint32_t)

       _mm512_store_epi32 (auxData+(Sup-x), MASKFAKE_AVX512);       //Padding (*********CALCULAR EL TAMAÑO EXACTO DEL TROZO RESTANTE**********)
 
       data = _mm512_load_epi32 ((void*) (auxData)) ;		
       outConf = _mm512_conflict_epi32(data); 
       VPI = VPIgenerate (outConf);
       mymask = VLUgenerateMASK(outConf);	
       auxPS= _mm512_i32gather_epi32 (data, (void *) ExPsum, 4);          //3. Gather from Psum al the indexes in VPI  
       auxAdd = _mm512_add_epi32 (auxPS, VPI);        			  //4 Adds VPI and auxPSum, these are the offsets

       Suff = _mm512_load_epi32 ((void*) (SortedArr+x)) ;                //5. Read the suffixes to sort
       _mm512_i32scatter_epi32 ((void *) NewSortedArr, auxAdd,Suff, 4) ; //6. Creates the partial order  of suffixes

	if (auxData) free(auxData);*/

 }//End of parallel region
       #ifdef DEBUG_PR //Partial Radix         
	  for (int i=Inf; i<Sup-1;i++){ 
	      //if (SortedArr[i-1]>SortedArr[i]) printf ("Error of sorting in SortedArr[%d]= %d\n ",i, SortedArr[i]) ; 
	      if (In[NewSortedArr[i]]>In[NewSortedArr[i+1]]){
	 printf (" Mal orden ParRadix pos[%u], suf=%u ValNewSorted=%u, suf+1=%u, - ValNewSorted[+1]=%u, \n ",i, NewSortedArr[i],In[NewSortedArr[i]], NewSortedArr[i+1], In[NewSortedArr[i+1]]) ; 
		int c = getchar ();	
		}	      
	  }
	 /*FILE *outFil=fopen ("InputParRad.txt", "w"); 
    	   for (int64_t m=0; m<sizeInArr; m++){
		fprintf(outFil, " %u\n ", In[m]);
		printf(" %u ", In[NewSortedArr[m]]);  
	}	
	fclose(outFil);*/

	 //outFil=fopen ("SufParRad.txt", "w"); 
    	//   for (int64_t m=0; m<sizeInArr; m++){		fprintf(outFil, " %u\n ", NewSortedArr[m]);	}	
	//fclose(outFil); int c = getchar();	
       #endif
       #ifdef PRINT_PR //Partial Radix         
	  FILE *outFile=fopen ("PartialRadixOutPut.txt", "w"); 
	  for (int i=Inf; i<Sup;++i){ 
	      fprintf(outFile,"Input[%u]=%u, Input[%u]==%u Old Suf %u \n",  NewSortedArr[i], In[NewSortedArr[i]], i,In[i], SortedArr[i] );
	  }
	  if (outFile) fclose(outFile);
       #endif

}//end of PartialRadixSA




void inline  SelectValidCharMIMD (uint32_t *InputText, uint32_t Size, uint32_t currentIter, uint32_t *Indexes, uint32_t *OutInt, int64_t Inf, int64_t Sup, int Nthreads ){// Indexes = SortedSuffixes, SizeInput is exact

   uint32_t ChunkSize=(uint32_t) ((Sup-Inf)/Nthreads)+ADJUST;
   #pragma omp parallel  num_threads(Nthreads) shared(ChunkSize) //firstprivate(Res,Displacement)
   { 
		
	int tid = omp_get_thread_num(); //Pido Mi Id,
  	uint32_t Inicio= (tid*ChunkSize)+Inf;
	uint32_t Fin = MIN(Sup-MRL , (Inicio + ChunkSize-MRL));  // considerr el caso del ultimo chunk
	if (ChunkSize<MRL) Fin =0;	

        	//int32_t It, Displace = (int32_t) (SizeInput-1)-(int32_t)currentIter;          , in the MSC approach is just the currentIter
	uint32_t *auxSuf, It; 
        __m512i Suffix, TextIndex, Chars, Displ;  
	auxSuf=(Indexes+currentIter);        
	Displ = _mm512_set1_epi32(currentIter);             //Displacement value 
	//printf ("Soy SVC PAR %d, Inicio =%lu, Fin = %lu, It= %lu, Sup = %lu\n",tid,Inicio,Fin, It, Sup);fflush(stdout);
        for (It=Inicio;It<Fin;It+=MRL){            	 

             Suffix = _mm512_load_epi32 ((void*) (Indexes+It)) ;                   //2. Load contiguos suffixes
       
             TextIndex= _mm512_add_epi32(Suffix, Displ) ;                          //3. Sum those sufixes and the Displace Value

             Chars = _mm512_i32gather_epi32 (TextIndex, (void *) InputText, 4);    //4.- Bring chars from text 
             _mm512_store_epi32 (OutInt+It,Chars);                                 //5.- Store them in the OutPut

       	    /*#ifdef PRINT_SVC_SIMD
               printf ("iIt %i  Displace %i \n", It,  Displace);fflush(stdout);
	       printf ("It  %i Displace %i \n", It, Displace);fflush(stdout);
               print_vector((void*)&Suffix,    "        Suffix", "AVX512", "uint32_t", "DEC");fflush(stdout);
               print_vector((void*)&TextIndex, "     TextIndex", "AVX512", "uint32_t", "DEC");fflush(stdout);//}
               print_vector((void*)&TextIndex, "    InputIndex", "AVX512", "uint32_t", "DEC");fflush(stdout);
               print_vector((void*)&Chars,     "        Values", "AVX512", "uint32_t", "DEC");fflush(stdout);
               //int c = getchar();
            #endif*/


        }//end of for It=0   
	//printf ("Soy %d SEQ VC SIMD + OPT\n",tid);fflush(stdout);
	uint32_t ValidChar;
	Fin= MIN(Sup, (Inicio + ChunkSize)); // OJO(Si llega a presentar error habilitar este:) if (tid==(Nthreads-1))  Fin=Sup ;
	//printf ("Soy SVC SEQ %d, Inicio =%lu, Fin = %lu, It= %lu, Sup = %lu\n",tid,Inicio,Fin, It, Sup);fflush(stdout);
        for (uint32_t auxIt=It;auxIt<Fin;auxIt++){   /*OJO Debe ser menor estricto*/
		ValidChar = Indexes[auxIt] + currentIter;
		if(!(ValidChar >= Size)){       //PAdding 
		    OutInt [auxIt] = InputText [  ValidChar ] ;  
		}else{
		    OutInt[auxIt] = HIGHESTCHAR;		//reevaluar, Caracter Invalido. Debe ser cero o debe ser un numero muy grande para que no afecte el prefix sum
		}
	}

	 //printf ("Soy %d SVC ITERATIVO OK\n",tid);fflush(stdout);
   }//end pragma
	 
        #ifdef DEBUG_SVC
           //printf ("\nSTART DEBUG SVC \n");fflush(stdout);
	   for (uint32_t It=Inf;It<Sup;++It){  //Para cada uno d elos caracterres validos    
		//printf("START \n" );	
		uint32_t aux, ValidChar = Indexes[It] + currentIter;
		if(!(ValidChar >= Size)){       //PAdding 
		//printf("Validchar if %u \n", ValidChar );
		    aux = InputText [  ValidChar ] ;  
		}else{
 		//printf("Validchar ese %u \n", ValidChar );
		    aux = HIGHESTCHAR;		//reevaluar, Caracter Invalido. Debe ser cero o debe ser un numero muy grande para que no afecte el prefix sum
		}
		if (OutInt[It]!=aux) {
			//printf("\nSVC_ERROR Iter %u Elem %"PRIu32"     Deseada=  %"PRIu32"      Calculada %"PRIu32"       ValidChar = %"PRIu32"  \n", currentIter,It, aux, OutInt[It], ValidChar  ); 
			int c = getchar ();	
		} //else printf("\nSVC Deseada=  %"PRIu32"      Calculada %"PRIu32"       ValidChar = %"PRIu32"  \n", aux, OutInt[It], ValidChar  ); //printf ("SVC OK");
		//printf("END \n");	
	   }
	    //printf ("Si no vio un mensaje de error la depuracion SVC fue perfecta\n" );
            //int c = getchar();
       #endif
 
        
}//EndSelectValidChars



void inline GetHistogramMIMD (uint32_t *InputArr, uint32_t sizeInArr, uint32_t *Hist_, int64_t Inf, int64_t Sup, int Nthreads){ //TODOS LOS VALORES E INF es inclusivo y SUP es exclusivo
   uint32_t ChunkSize=(uint32_t) ((Sup-Inf)/Nthreads)+ADJUST;

   uint32_t sizeHi=SIZEHIST*sizeof(uint32_t);
   //printf("\nSTART Inf %u Sup %u Chun %u \n", Inf, Sup, ChunkSize);
   //Meter el calculo cubetas privadas Hist y luego totalizar sobre Hist_
   #pragma omp parallel  num_threads(Nthreads) shared(ChunkSize,InputArr,Hist_) //firstprivate(Res,Displacement)
   { 
	uint32_t *Hist, x; //Local Histogram,   
   	int succP_= posix_memalign ((void **) &Hist, 64, sizeHi);
	if (succP_!=0){        printf ("Error reserving memory for Private Histogram");       	exit(0);    } 
	memset(Hist,0,sizeHi);

	int tid = omp_get_thread_num(); 
  	uint32_t Inicio= (tid*ChunkSize)+Inf;
	uint32_t Fin = MIN(Sup -MRL, (Inicio + ChunkSize)-MRL);  // considerr el caso del ultimo chunk
	if (ChunkSize<MRL) Fin =0;			//OPT Esta linea se puede quitar siempre que la est de dato del histograma se llene con un numero muy grande 255 x ejemplo. 
	//El problema es como cuando hay menos de 16 datos el SIMD puede generar caluclo erroneo 
	
	//printf ("soy %d Par, Inicio %d Fin %d \n", tid , Inicio, Fin);

        uint32_t *auxInput;  
	__m512i data,outConf, VPI, VLU, VPI1, currentData, _Add;  
	__mmask16 mymask, mymask_, mymask2;

	for (x=Inicio; x<Fin;x+=MRL){ 
	   //printf ("x=%u ",x); fflush(stdout);
		auxInput=(InputArr+x);                
		data = _mm512_load_epi32 ((void*) auxInput) ;

		outConf = _mm512_conflict_epi32(data);        
		VPI= VPIgenerate(outConf);
		mymask = VLUgenerateMASK(outConf)^mask1 ; 
		VPI1 = _mm512_add_epi32(MASKONE_AVX512,VPI);   

		currentData = _mm512_mask_i32gather_epi32 (MASKZERO_AVX512, mymask,data , (void const *)Hist,4); //bring indexes from GLOBALHist in memory
		//si el SAC es únicamente para genómica se pueden eliminar todas las cubetas y hacer el cálculo del histograma directamente sobre Registros
		_Add = _mm512_add_epi32 (currentData, VPI1);                         //Sum The local Histogram (just calculated) with the previously existing 
		_mm512_mask_i32scatter_epi32 ((void *)Hist, mymask,data , _Add, 4) ; //Update Global Hist, needs the mask for the one that has been added

	}//end for x=0

	Fin = MIN(Sup, (Inicio + ChunkSize));
	uint32_t auxIt;
	//printf ("soy %d en SEQ, x %d Fin %d \n", tid , x, Fin);
	for ( auxIt=x;auxIt<Fin;auxIt++){ 
		//printf ("soy %d en bloque B, auxIt %d Fin %d\n", tid , auxIt, Fin);
		Hist[InputArr[auxIt]]++;
	}


/*
   //shared(ChunkSize,InputArr,Hist_) quitado 20-10 -18
	x=Inicio;
	Fin = MIN(Sup, (Inicio + ChunkSize));  //f considerr el caso del ultimo chunk

	if (ChunkSize>MRL){ //Fin =0;	//El problema es como cuando hay menos de 16 datos el SIMD puede generar caluclo erroneo 
		Fin=Fin-MRL;
	}//endif

	//Fin = MIN(Sup, (Inicio + ChunkSize));

	for ( auxIt=x;auxIt<Fin;auxIt++){ 
		
		Hist[InputArr[auxIt]]++;
	}

	//Totalizando los histogramas locales Hist, en el Histograma resultado Hist_
        #pragma omp critical //(Hist)
        for(i = 0; i < 255; i+=MRL) { //La parte SIMD
	
	     //auxOutput=(Hist_+i);
        }
}//end of GetHistogram
*/



	uint32_t i, *auxBucket, *auxOutput, k;
	 __m512i RegBucket;
	//Totalizando los histogramas locales Hist, en el Histograma resultado Hist_
        #pragma omp critical (HIST)
        for(i = 0; i < 256/*(SIZEHIST-MRL)*/; i+=MRL) { //La parte SIMD
	     auxInput=(Hist_+i);   //Leer del global
	     data = _mm512_load_epi32 ((void*) (auxInput)) ;

	     auxBucket=(Hist+i);//local
             RegBucket= _mm512_load_epi32 ((void*) auxBucket) ;//Leer de la cubeta

	     _Add = _mm512_add_epi32 (data, RegBucket); //sumar

	     //auxOutput=(Hist_+i);
	     _mm512_store_epi32 (auxInput/*auxOutput*/, _Add); //devolver al global
		
        }
	/*#pragma omp critical
       	for(uint32_t j = i; j <= SIZEHIST-MRL; j++) { //La parte Secuencial
             Hist_[j] += Hist[j];
        }*/
	
	if (Hist) free(Hist);
  }//end parallel region
  
   #ifdef DEBUG_H         
      //for (int32_t i=0; i<sizeInArr;++i)  if( InputArr[i]>0)printf ("Input[%d]= %d  e Hist = %d \n",i, InputArr[i], Hist[InputArr[i]]) ; /*int c=getchar();*/
      int TestH[SIZEHIST];
      for (int32_t i=0; i<256;++i)   TestH[i]=0;   
      for (int32_t i=Inf; i<Sup;++i)  TestH[ InputArr[i]  ]++;
      for (int32_t i=0; i<256;++i)   
 	    if (TestH[i]!=Hist_[i])  { printf ("HIST_ERROR Inf %u Sup%u CalculatedH[%d]= %d ERROR vs DesiredHist[%d]= %d \n",Inf, Sup,i, Hist_[i],i,TestH[i]) ; /*exit(0);*/ }
   #endif

}//end of GetHistogram
















void inline  OPTPartialRadixSAC_SIMD (uint32_t *InputArr, uint32_t sizeInArr, uint32_t *Hist_, int64_t Inf, int64_t Sup, int Nthreads, uint32_t *In /*putExp*/, uint32_t *SortedArr , uint32_t *NewSortedArr,uint32_t *ExPsum){   //Expsum puede pasar al  interno.

 uint32_t ChunkSize=(uint32_t) ((Sup-Inf)/Nthreads)+ADJUST;

   uint32_t sizeHi=SIZEHIST*sizeof(uint32_t);
   //printf("\nPAR HIST Inf %u Sup %u Chun %u \n", Inf, Sup, ChunkSize);
   //Meter el calculo cubetas privadas Hist y luego totalizar sobre Hist_
   #pragma omp parallel  num_threads(Nthreads) shared(ChunkSize,Hist_) //private(SortedArr)
   { 
	uint32_t *Hist, x; //Local Histogram,   
   	int succP_= posix_memalign ((void **) &Hist, 64, sizeHi);
	if (succP_!=0){        printf ("Error reserving memory for Private Histogram");       	exit(0);    } 
	memset(Hist,0,sizeHi);

	int tid = omp_get_thread_num(); 
  	uint32_t Inicio= (tid*ChunkSize)+Inf;
	uint32_t Fin = MIN(Sup -MRL, (Inicio + ChunkSize)-MRL);  // considerr el caso del ultimo chunk
	if (ChunkSize<MRL) Fin =0;			//OPT Esta linea se puede quitar siempre que la est de dato del histograma se llene con un numero muy grande 255 x ejemplo. 
	//El problema es como cuando hay menos de 16 datos el SIMD puede generar caluclo erroneo 
	
	//printf ("soy %d Par, Inicio %d Fin %d \n", tid , Inicio, Fin);

        uint32_t *auxInput;  
	__m512i data,outConf, VPI, VLU, VPI1, currentData, _Add;  
	__mmask16 mymask, mymask_, mymask2;

	for (x=Inicio; x<Fin;x+=MRL){ 
	   //printf ("x=%u ",x); fflush(stdout);
	    	//auxInput=(InputArr+x);                
		data = _mm512_load_epi32 ((void*) (InputArr+x)) ;

		outConf = _mm512_conflict_epi32(data);        
		VPI= VPIgenerate(outConf);
		mymask = VLUgenerateMASK(outConf)^mask1 ; 
		VPI1 = _mm512_add_epi32(MASKONE_AVX512,VPI);   

		currentData = _mm512_mask_i32gather_epi32 (MASKZERO_AVX512, mymask,data , (void const *)Hist,4); //bring indexes from GLOBALHist in memory
		//si el SAC es únicamente para genómica se pueden eliminar todas las cubetas y hacer el cálculo del histograma directamente sobre Registros
		_Add = _mm512_add_epi32 (currentData, VPI1);                         //Sum The local Histogram (just calculated) with the previously existing 
		_mm512_mask_i32scatter_epi32 ((void *)Hist, mymask,data , _Add, 4) ; //Update Global Hist, needs the mask for the one that has been added

	}//end for x=0

	Fin = MIN(Sup, (Inicio + ChunkSize));
	uint32_t auxIt;
	//printf ("PAR HIST soy %d en SEQ, x %d Fin %d \n", tid , x, Fin);
	for ( auxIt=x;auxIt<Fin;auxIt++){ 
		//printf ("soy %d en bloque B, auxIt %d Fin %d\n", tid , auxIt, Fin);
		//printf("InputArr [%u]=%u\n",auxIt,InputArr[auxIt] );
		Hist[InputArr[auxIt]]++; 
	}



	uint32_t i, *auxBucket, *auxOutput, k;
	 __m512i RegBucket;
	//Totalizando los histogramas locales Hist, en el Histograma resultado Hist_
        #pragma omp critical (HIST)
        for(i = 0; i < 256/*(SIZEHIST-MRL)*/; i+=MRL) { //La parte SIMD
	     auxInput=(Hist_+i);   //Leer del global
	     data = _mm512_load_epi32 ((void*) (auxInput)) ;

             RegBucket= _mm512_load_epi32 ((void*) (Hist+i)) ;//Leer de la cubeta

	     _Add = _mm512_add_epi32 (data, RegBucket); //sumar

	     _mm512_store_epi32 (auxInput, _Add); //devolver al global
		
        }
 	#pragma omp barrier

   #ifdef DEBUG_H   
     #pragma omp single
	{     
      //for (int32_t i=0; i<sizeInArr;++i)  if( InputArr[i]>0)printf ("Input[%d]= %d  e Hist = %d \n",i, InputArr[i], Hist[InputArr[i]]) ; /*int c=getchar();*/
      int TestH[SIZEHIST];
      for (int32_t i=0; i<256;++i)   TestH[i]=0;   
      for (int32_t i=Inf; i<Sup;++i)  TestH[ InputArr[i]  ]++;
      for (int32_t i=0; i<256;++i)   
     if (TestH[i]!=Hist_[i])  { printf ("HIST_ERROR OPT Inf %u Sup%u CalculatedH[%d]= %d ERROR vs DesiredHist[%d]= %d \n",Inf, Sup,i, Hist_[i],i,TestH[i]) ; int c=getchar();/*exit(0);*/ }
	}
   #endif
		
	 #pragma omp single
		Psum (Hist_, ExPsum);
		//ExcPsum (Hist_, ExPsum);





	for(int32_t cur_t = Nthreads - 1; cur_t >= 0; cur_t--) {
                if(cur_t == tid) {
                    for(i = 0; i < 256; i++) {
				//if (ExPsum[i] <Hist[i]) printf ("algo negativo en la pos %u , Ex=%u, Hist%u  \n", i,ExPsum[i],Hist[i] );
                        ExPsum[i] -= Hist[i];
                        Hist[i] = ExPsum[i];
                    }
                } else { //just do barrier
                    #pragma omp barrier
                }
 
            }

//Partial RAdix NEW PARALLEL , el codigo quedo un poco ilegible por la reutilizacion de regitsros para evitar se saturaran
	Fin = MIN(Sup-MRL , (Inicio + ChunkSize-MRL));  // Ver abajo el procesamiento del bloque remanente
	if (ChunkSize<MRL) Fin =0;			//OPT Esta linea se puede quitar siempre que el grupo sea mayor a 17*NT= 1500 en nuestro caso	
	__m512i Suff;
	uint32_t *auxNSArr=(NewSortedArr+Inf);
	//printf("\nPAR RADIX PAR Inf %u Sup %u Chun %u \n", Inf, Sup, ChunkSize);
	for (x=Inicio; x<Fin;x+=MRL){

		//auxOutput=;
		data = _mm512_load_epi32 ((void*) (InputArr+x)) ; //lea OutChar

		RegBucket=_mm512_i32gather_epi32 (data, (void *) Hist, 4);	//Traiga de Hist un gather cn lo que habia en outchar (=EXPSUM)

		
		outConf = _mm512_conflict_epi32(data); //Remplazar con la lectura de un arreglo compartido de VLU Y VPI .
		VPI = VPIgenerate (outConf);
		mymask = VLUgenerateMASK(outConf)^mask1;

		_Add=_mm512_add_epi32 (RegBucket,  VPI);        	//4 Adds VPI and auxPrSum, these are the offsets

		Suff = _mm512_load_epi32 ((void*) (SortedArr+x)) ;       //5. Suffixes to sort
		_mm512_i32scatter_epi32 ((void *) auxNSArr, _Add,Suff, 4); //6. Place them


                outConf = _mm512_add_epi32 (_Add, MASKONE_AVX512);                       //Adds the current Values of VPI and Corresponding prefix sum positions
	        _mm512_mask_i32scatter_epi32 ((void *)Hist, mymask,data , outConf, 4) ;  //Updates PrefixSum (HIst) //Mask is needed to avoid collisions cause data (indexing here) could have repeated 
											//elements, updating only the according to the last appeareance	
	}//end for x
//PR SEQ
	Fin= MIN(Sup, (Inicio + ChunkSize)); 
	//#pragma omp for schedule (static)	//Realmente hay muy pocos elementos
	//printf("\nPAR RADIX SEQ Inf %i Sup %i Chun %u. x= %u, Fin=%u  \n", Inf, Sup, ChunkSize, x, Fin);
	for (uint32_t auxIt=x/*Inicio*/;auxIt<Fin;auxIt++){
		NewSortedArr[  (Hist[ InputArr[auxIt] ]++) +Inf ]= SortedArr[auxIt];
		} 

	
	if (Hist) free(Hist);
  }//end parallel region OPT

	 #ifdef DEBUG_PR //Partial Radix         
	  for (int i=Inf; i<Sup-1;i++){ 
	      //if (SortedArr[i-1]>SortedArr[i]) printf ("Error of sorting in SortedArr[%d]= %d\n ",i, SortedArr[i]) ; 
	      if (In[NewSortedArr[i]]>In[NewSortedArr[i+1]]){
	        printf (" Mal orden ParRadix pos[%u], suf=%u ValNewSorted=%u, suf+1=%u, - ValNewSorted[+1]=%u, \n ",i, NewSortedArr[i],In[NewSortedArr[i]], NewSortedArr[i+1], In[NewSortedArr[i+1]]) ; 
		int c = getchar ();	
		}	      
	  }
       #endif

};//endOPT

//Version optimized for SACAS. For a general vision of any string checks the notebook. 
//Radixsort sorts starting from the char at the most right position to the left. An additional single character is incorporated  at every iteration. 
//It depends on selecting previously the correct characters to be ordered in every iteration. Sorts Suffixes not chars.Handles any size of input
void inline  PartialRadixSA (uint32_t *InputArr, uint32_t sizeInArr, uint32_t *ExPsum, uint32_t *SortedArr , uint32_t *NewSortedArr, uint32_t *VPI_Arr, uint32_t *VLU_Arr){
//El arreglo de datos inicia en CERO desde un punto de vista local. El arreglo de Sufijos inicia exactamente en la posicion de los elementos a ordenar y  hay paralelismo entre el arreglo de datos y el de sufijos

    __m512i VPI, VLU, VPI1, auxPS, auxAdd,Add1,Suff,data;//, outConf; 
    __mmask16 mymask;

     for (uint32_t x=0; x<sizeInArr;x+=16){ 

        data = _mm512_load_epi32 ((void*) (InputArr+x)) ;

        //VPI = _mm512_load_epi32 ( (VPI_Arr+x));    //Both are parallel to data 
        //VLU = _mm512_load_epi32 ( (VLU_Arr+x));
        // Only if you find out that calculating is cheaper than storing/loading
        __m512i outConf = _mm512_conflict_epi32(data); 
        VPI = VPIgenerate (outConf);
        mymask = VLUgenerateMASK(outConf)^mask1 ;
        
        //mymask = _mm512_cmpeq_epu32_mask (VLU,MASKONE_AVX512); //2. Compress VLU

        //3. Gather from Psum al the indexes in VPI  
        auxPS= _mm512_i32gather_epi32 (data, (void *) ExPsum, 4);

        //4 Adds VPI and auxPrSum, these are the offsets
        auxAdd = _mm512_add_epi32 (auxPS, VPI);

        Suff = _mm512_load_epi32 ((void*) (SortedArr+x)) ;               //Suffixes to sort
 	_mm512_i32scatter_epi32 ((void *) NewSortedArr, auxAdd,Suff, 4) ;//Creates the partial order  of suffixes

        Add1 = _mm512_add_epi32 (auxAdd, MASKONE_AVX512);                       //Adds the current Values of VPI and Corresponding prefix sum positions
        _mm512_mask_i32scatter_epi32 ((void *)ExPsum, mymask,data , Add1, 4) ;  //Updates PrefixSum. //Mask is needed to avoid collisions cause data (indexing here) could have repeated elements, updating only the according to the last appeareance

        #ifdef PRINT_PR          
           printf ("*********Internal Partial Sorting Iteration***** %u\n ", x/16);     
	   print_vector((void*)&data,     "      Data", "AVX512", "uint32_t", "DEC");
           //print_vector((void*)&outConf, "outCnf     ", "AVX512", "uint32_t", "DEC");
           //print_vector((void*)&VLU,     "        VLU", "AVX512", "uint32_t", "DEC");            
	   //print_vector((void*)&VPI,     "        VPI", "AVX512", "uint32_t", "DEC");
           printf("mymask Hexa = %x \n",mymask);
           print_vector((void*)&auxPS,   "DatFromPsum", "AVX512", "uint32_t", "DEC");          
	   print_vector((void*)&auxAdd,  "Offsets    ", "AVX512", "uint32_t", "DEC");          
           print_vector((void*)&Add1,    "UpdatedPsum", "AVX512", "uint32_t", "DEC");
	   print_vector((void*)&Suff,    "Sufijos    ", "AVX512", "uint32_t", "DEC");          
           //print_vector((void*)&data,   "TargtIndex ", "AVX512", "uint32_t", "DEC");
           printf ("\n");
       	   //for (int i=0; i<sizeInArr;++i)           	printf ("SortedArr[%d]= %d,\n ",i, SortedArr[i]) ;
        #endif

  }//end of for x=0

  #ifdef PRINT_PR //Partial Radix         
      for (int i=0; i<sizeInArr;++i){ 
   	 //if (SortedArr[i-1]>SortedArr[i]) printf ("Error of sorting in SortedArr[%d]= %d\n ",i, SortedArr[i]) ; 
   	printf (" NewSortedArr[%d]= %d \t ",i, NewSortedArr[i]) ; 
      }
   	
  #endif

}//end of PartialRadixSA


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


