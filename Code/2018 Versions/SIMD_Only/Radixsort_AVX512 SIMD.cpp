#include <iostream>
//#include <cstdlib>
//#include <stdint.h>
#include <fstream> 
#include <string.h>
#include <immintrin.h>
#include <chrono>
#include <omp.h>

#define SIZEPADDING 32 
#define MRL 16           //Maximum Register Length
#define TAM_ENTRADA 355018//200018//99951 
#define SIZEHIST 66018 //(This is for pack 2) 2114999//for A*16384 -299800(A*1024) //299280 //=299264+16 //272 //Size of Histogram, Made this way to protect the Psum from errors resulting from the fake values
#define MAXSIZEINPUT 0xFFFFFFFF
#define STEP 2

//Constant for Debugging
#define DEBUG //Debug main 
//#define CHECK_Point  
//#define  DEBUG_PCNT_OPT
//#define DEBUG_EXPAND  
//#define DEBUG_H //Debugging Histogram
//#define DEBUG_HM //DEbugging Histogram to the main level
//#define DEBUG_OPC //DEbugging OutputChars
//#define DEBUG_PC //DEbugging Pop Count
//#define DEBUG_MCpy 
#define DEBUG_IS //DebugInitiate Suffixes  
//#define DEBUG_PS
//#define DEBUG_SVC //Debug Valid Char
//#define DEBUG_SVC_SIMD //Debug Valid Char SIMD
//#define DEBUG_PR //Partial Radix Debug         
//#define DEBUG_GR //Global Radix Debug         


//Constant for pronting values while Debugging
//#define PRINT //Debug the main  
//#define PRINT_FINAL_SORT //Debug the main  
//#define PRINT_H//Debugging all Histogram
//#define PRINT_HM //DEbugging Histogram to the main level
//#define PRINT_OPC //DEbugging OutputChars
//#define PRINT_PC //DEbugging Pop Count
//#define PRINT_Cpy 
//#define PRINT_IS //DebugInitiate Suffixes  
//#define PRINT_PS
//#define PRINT_SVC //Debug Valid Char
//#define PRINT_SVC_SIMD //Debug Valid Chari SIMD
//#define PRINT_PR //Partial Radix Debug         
//#define PRINT_GR //Global Radix Debug         
//#define PRINT_VLUVPI

   //Constant AVX_512 Registers  
   __m512i MASKZERO_AVX512  = _mm512_set_epi32(0x0, 0x0, 0x0,0x0,0x0,0x0,0x0,0x0,0x0, 0x0, 0x0,0x0,0x0,0x0,0x0,0x0);
   __m512i MASKONE_AVX512   = _mm512_set_epi32(0x1, 0x1, 0x1,0x1,0x1,0x1,0x1,0x1,0x1, 0x1, 0x1,0x1,0x1,0x1,0x1,0x1);

   //THIS IS FOR Pack 2 65984. //, 299776 = 0x49300. Para 2 114 688 = 20446E i, se ajusta segun tamaño del histogram y el exceso de 16a. 
   __m512i MASKFAKE_AVX512   = _mm512_set_epi32(0x101C0, 0x101C0, 0x101C0,0x101C0,0x101C0,0x101C0,0x101C0,0x101C0,0x101C0, 0x101C0, 0x101C0,0x101C0,0x101C0,0x101C0,0x101C0,0x101C0);  //FOR PADDING values Pack 2 

   __m512i MASK_AVX512_1to16 = _mm512_set_epi32(16,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1);          //Used  for Initializing Suffixes and SVC
   __m512i MASK_AVX512_16    = _mm512_set_epi32(16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16); //Used  for Initializing Suffixes and SVC


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

using namespace std;
using namespace std::chrono;
using timer = std::chrono::high_resolution_clock;


//PROTOTYPES
void print_vector(void *vec_ptr, std::string comment, std::string simdtype, std::string datatype, std::string format); 
void inline RadixSORT_SA_AVX512 (uint32_t SizeIn32, uint32_t Size32Pad, uint32_t *InputArr, uint32_t sizeInArr, uint32_t *SPos ); 
void inline SelectValidCharsSA (uint32_t *InputText, uint32_t SizePadded, uint32_t currentIter, uint32_t *Indexes, uint32_t SizeInput, uint32_t *OutInt);//Pass the size-1, Indexes = SortedPos
void inline InitializeSuffixSIMD (uint32_t SizeInput, uint32_t *Suffixes);
void inline InitializeOutPutChars(uint32_t *aux, uint32_t size) ;
//void inline GetHistogramAVX512 (__m512i data , uint32_t *Hist, uint32_t *VPI_Arr, uint32_t *VLU_Arr, uint32_t PosVPI);
void inline Expand_Pck2 (uint8_t *InputArr8, uint32_t *InputArr32 , uint32_t sizeInArr8); 
__m512i inline __attribute__((always_inline)) VPIgenerate(__m512i In);
static inline __attribute__((always_inline)) __m512i _mm512_ExPrefixSum_epi32(__m512i val) ; 
void inline GetHistogram (uint32_t *InputArr, uint32_t sizeInArr, uint32_t *Hist, uint32_t *VPI_Arr, uint32_t *VLU_Arr);
inline __m512i VLUgenerate(__m512i Conf);
void inline ExcPsum (uint32_t *Hist, uint32_t *ExPsum);

inline __m512i VLUgenerate(__m512i Conf){
   __mmask16 mask=(__mmask16)_mm512_reduce_or_epi32(Conf);        // Builds a mask with all of the elements that are 1 o more 
   return( _mm512_mask_set1_epi32(MASKONE_AVX512, mask,0));       //Puts Zero in all of the positions previously repeated, according to 1's in the mask  
}   


void inline GetHistogram (uint32_t *InputArr, uint32_t sizeInArr, uint32_t *Hist, uint32_t *VPI_Arr, uint32_t *VLU_Arr){

   uint32_t *auxInput;  
   __m512i data,outConf, VPI, VLU, VPI1, currentData, _Add;  
   __mmask16 mymask, mymask_, mymask2;

   for (uint32_t x=0; x<sizeInArr;x+=16){ 

        auxInput=(InputArr+x);                
        data = _mm512_load_epi32 ((void*) auxInput) ;

        outConf = _mm512_conflict_epi32(data);        
        VPI= VPIgenerate(outConf);
        VLU = VLUgenerate(outConf);
        VPI1 = _mm512_add_epi32(MASKONE_AVX512,VPI);   


        /*****STORES VLU and VPI to load them when applying RadixSort********/
        _mm512_store_epi32 ((VPI_Arr+x),VPI);
        _mm512_store_epi32 ((VLU_Arr+x),VLU);

       /*  Escenario: Vengo calculando VLU en dos operaciones: una reduccion y una compresion. Ahora lo puedo hacer todo en una reduccion y un or exclusivo, lo cual TEORICAMENTE Es mas economico.  
        Esto me permite , al menos TEORICAMENTE otra optimizacion, ya guardar el VLU no es guardar un registro de 512 en memoria, sino guardar una mascara de 16 bits. El problema con esto es, que guardaria un 
        VLU por iteracion, y esto no es una operacion de registros simd sino un acceso a memoria. Ademas requeriria operaciones adicionales para administrar la posicion de almacenamiento en este arreglo */
       
        mymask = _mm512_cmpeq_epu32_mask (VLU,MASKONE_AVX512);  //colocar operación de VLU optimizada               //Calculates the mask compressing VLU
        //mymask_= VLUgenerateMASK(outConf);//optimized
        //mymask2 = _mm512_testn_epi32_mask(outConf, MASKZERO_AVX512);
        //printf ("******** MASCARAS  ********* (deseada) mymaskCMPEQ %x vs maskReduceOr %x  Xor %x, nand %x \n", mymask, mymask_, mymask_^mask1, mymask2);
        //char c=getchar();


        currentData = _mm512_mask_i32gather_epi32 (MASKZERO_AVX512, mymask,data , (void const *)Hist,4); //bring indexes from GLOBALHist in memory
        //si el SAC es únicamente para genómica se pueden eliminar todas las cubetas y hacer el cálculo del histograma directamente sobre los registros
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
      for (int32_t i=0; i<SIZEHIST;++i)   TestH[i]=0;   
      for (int32_t i=0; i<sizeInArr;++i)  TestH[ InputArr[i]  ]++;
      for (int32_t i=0; i<sizeInArr;++i)   
 	    if (TestH[i]!=Hist[i])  { printf ("CalculatedH[%d]= %d ERROR vs DesiredHist[%d]= %d \n",i, Hist[i],i,TestH[i]) ; }
   #endif
   #ifdef PRINT_H         
      //for (int32_t i=0; i<SIZEINST;i+=6) printf ("Hist[%d]= %d \tHist[%d]= %d \tHist[%d]= %d\tHist[%d]= %d \t Hist[%d]= %d \t Hist[%d]= %d  \n",i, Hist[i],i+1,Hist[i+1],i+2,Hist[i+2],i+3, Hist[i+3] ,i+4, Hist[i+4],i+5, Hist[i+5] ) ;
   #endif

}//end of GetHistogram




void inline ExcPsum (uint32_t *Hist, uint32_t *ExPsum){          //GLOBAL PREFIXSUM: Calculates ExPsum over the input Array, multiple of 16
         uint32_t *auxH, *aux1;
         __m512i Add, dat, LocExPSum ; 
         __m512i auxAdd = MASKZERO_AVX512;                       //Initialize the aux register that helps in calculating the global Psum

         for (int x=0; x<SIZEHIST;x+=16){   
                auxH=(Hist+x);

		dat=_mm512_load_epi32 ((void *) auxH);   	 

 	        LocExPSum= _mm512_ExPrefixSum_epi32(dat);        //Calculate localReg ExPsum
 
                Add = _mm512_add_epi32 (LocExPSum,auxAdd );	 //Add with the previous iteration Psum and this is the real general Xpsum
 
               _mm512_store_epi32 ((void*) (ExPsum+x),Add) ;

               //Updates the auxiliarReg  for next iteration
               auxAdd = _mm512_add_epi32(Add, dat);
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
             //if (0!=ExPsum[0] ) printf ("Error ExPsumCalc[%d]= %d ERROR vs ExPsumDes[%d]= %u \n",0, ExPsum[0],0,auXPsum[0]) ;
             for (int x=1; x<SIZEHIST;++x){  
                auXPsum[x]=auXPsum[x-1]+Hist[x-1];
                if (auXPsum[x]!=ExPsum[x] ) printf ("Error ExPsumCalc[%d]= %d ERROR vs ExPsumDes[%d]= %u \n",x, ExPsum[x],x,auXPsum[x]) ;
             }	   
            //for (int i=0; i<256;i+=6) printf ("Hist[%d]= %d \tHist[%d]= %d \tHist[%d]= %d\tHist[%d]= %d \t Hist[%d]= %d \t Hist[%d]= %d  \n",i, Hist[i],i+1,Hist[i+1],i+2,Hist[i+2],i+3, Hist[i+3] ,i+4, Hist[i+4],i+5, Hist[i+5] ) ;//Verifico qionado             
         #endif 

} //end of ExcPsum  


//Version optimized for SACAS. For a general vision of any string checks the notebook. 
//Radixsort sorts starting from the char at the most right position to the left. An additional single character is incorporated  at every iteration. 
//It depends on selecting previously the correct characters to be ordered in every iteration. Sorts Suffixes not chars.Handles any size of input
void inline PartialRadixSA (uint32_t *InputArr, uint32_t sizeInArr, uint32_t *ExPsum, uint32_t *SortedArr , uint32_t *NewSortedArr, uint32_t *VPI_Arr, uint32_t *VLU_Arr){
//El arreglo de datos inicia en CERO desde un punto de vista local. El arreglo de Sufijos inicia exactamente en la posicion de los elementos a ordenar y  hay paralelismo entre el arreglo de datos y el de sufijos

    __m512i VPI, VLU, VPI1, auxPS, auxAdd,Add1,Suff,data;//, outConf; 
    __mmask16 mymask;

     for (uint32_t x=0; x<sizeInArr;x+=16){ 

        data = _mm512_load_epi32 ((void*) (InputArr+x)) ;

        VPI = _mm512_load_epi32 ( (VPI_Arr+x));    //Both are parallel to data 
        VLU = _mm512_load_epi32 ( (VLU_Arr+x));
        // Only if you find out that calculating is cheaper than storing/loading
        /*__m512i outConf = _mm512_conflict_epi32(data); 
        VPI = VPIgenerate (outConf);
        VLU = VLUgenerate(outConf);*/
        
        mymask = _mm512_cmpeq_epu32_mask (VLU,MASKONE_AVX512); //2. Compress VLU

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



void inline SelectValidChars (uint32_t *InputText, uint32_t SizeInput, uint32_t currentIter, uint32_t *Indexes, uint32_t *OutInt, uint32_t *Hist_ ){// Indexes = SortedSuffixes, SizeInput is exact


       /*#ifdef PRINT_SVC_SIMD
	      for (int32_t I=0; I<SizeInput;++I)            //printf ("Incoming Indexes [%u]  = %u ",It, Indexes[It]);
	      printf("ValInp[%d] =  %d \t ", I, InputText [I] );fflush(stdout);printf ("\n");
       #endif*/

        int32_t It, Displace = (int32_t) (SizeInput-1)-(int32_t)currentIter;  
        uint32_t *auxSuf; 
        __m512i Suffix, TextIndex, Chars, Displ; 
        //cout<< " Displace "<< Displace<<"\n";
        auxSuf=(Indexes+Displace);        
       
 	Displ = _mm512_set1_epi32(Displace);             //Displacement value 
                             
        for (It=0;It<currentIter;It+=16){            //=

             Suffix = _mm512_load_epi32 ((void*) (auxSuf+It)) ;             //2. Load contiguos suffixes
       
             TextIndex= _mm512_add_epi32(Suffix, Displ) ;                   //3. Sum those sufixes and the Displace Value

             #ifdef PRINT_SVC_SIMD
              //if (It>10000){
   	        printf ("It  %i Displace %i \n", It, Displace);fflush(stdout);
                print_vector((void*)&Suffix,    "        Suffix", "AVX512", "uint32_t", "DEC");fflush(stdout);
                print_vector((void*)&TextIndex, "     TextIndex", "AVX512", "uint32_t", "DEC");fflush(stdout);//}
             #endif

             Chars = _mm512_i32gather_epi32 (TextIndex, (void *) InputText, 4);     //4.- Bring chars from text 
             _mm512_store_epi32 (OutInt+It,Chars);                                  //5.- Store them in the OutPut

       	    /*#ifdef PRINT_SVC_SIMD
               printf ("iIt %i  Displace %i \n", It,  Displace);fflush(stdout);
               print_vector((void*)&Suffix,    "        Suffix", "AVX512", "uint32_t", "DEC");fflush(stdout);
               print_vector((void*)&TextIndex, "    InputIndex", "AVX512", "uint32_t", "DEC");fflush(stdout);
               print_vector((void*)&Chars,     "        Values", "AVX512", "uint32_t", "DEC");fflush(stdout);
               //int c = getchar();
            #endif*/

            //uint32_t *Fake; 
            //GetHistogramAVX512 (Chars, Hist_, /*VPI_Arr*/Fake,Fake/* uint32_t *VLU_Arr*/, It);

        }//end of for It=0          

        _mm512_store_epi32 ((OutInt+currentIter+1),/**/MASKFAKE_AVX512); //1 more time, for the fakes values, igher so it does not affect the prefix sum 
        //    
 
         /*#ifdef PRINT_SVC_SIMD
	    OutInt [auxIt] = InputText [  Indexes[ Displace+auxIt]  +  Displace ] ;
            //printf("\nIter %u \t CurrIter %u \t SizeIn-It=  %u  \t  [Sufijo# %d ]\t TextPosition = %u  \t ", It, currentIter, SizeInput-It, Indexes[Displace+It] , Indexes[Displace+It] +Displace  ); 
            for (int32_t i=0;i<= currentIter; ++i) printf (" OutChars[%d]= %d ",i, OutInt[i]);
            cout << " CurrentIter " <<currentIter<<"\n";
            int c = getchar();
         #endif*/
}





/**************************************************************MAIN**********************************************************************
****************************************************************************************************************************************/
int32_t main( int32_t argc, char *argv[] )  {


    // ************************DECLARE AND RESERVE ALIGNED MEMORY FOR SORTING************************/
    uint32_t sizeInArr=TAM_ENTRADA;    //sizeInArr is the exact number of elements in Input array
    uint32_t sizePadded=sizeInArr+SIZEPADDING;     // porque removi el ceil de la division. Esto Es. 16 ceros delante de la data valida, luego 16 ceros al final
 
    unsigned char *InputArr;  /***************************CREATING ARTIFICIAL INPUT (DEVELOPMENT ONLY)*********************/
    int succP_= posix_memalign ((void **) &InputArr, 64, (sizePadded)*sizeof(unsigned char));
    if (succP_!=0){          printf ("Error reserving memory for InputArr");       exit(0);    }  
    srand(time(NULL)); 
    for (int i=0; i<sizeInArr;++i)  {  InputArr[i]= (unsigned char) ((rand() % 44)+46 ) ;        }//del 44  al 90  

    /**************************EXPANDING INPUT******************************************/
    uint32_t *InputArr32;        //Final 32 bits Input Array
    succP_= posix_memalign ((void **) &InputArr32, 64, (sizePadded)*sizeof(uint32_t));
    if (succP_!=0){         printf ("Error reserving memory for InputArr");      exit(0);     }  
    Expand_Pck2 (InputArr, InputArr32 , sizeInArr);

   //if (InputArr) free (InputArr);//UNCOMMENT in the final Version

    uint32_t *Suffixes; 
    succP_= posix_memalign ((void **) &Suffixes, 64, (sizePadded)*sizeof(uint32_t)); 
    if (succP_!=0){         printf ("Error reserving memory for Suffixes");            exit(0);    } 
    auto startLocal = std::chrono::high_resolution_clock::now();   
    RadixSORT_SA_AVX512 (sizeInArr, sizePadded, InputArr32, sizeInArr, Suffixes);
    auto stopLocal = std::chrono::high_resolution_clock::now();   
    cout << duration_cast<milliseconds>(stopLocal-startLocal).count() << " milliseconds in RadixSimd (cout).\n" << endl;
 
    #ifdef DEBUG  
        int a=0;  char SizeInString[16];
        sprintf(SizeInString, "OutRadixSort512_%d.txt", sizeInArr);
        
 	FILE *outFile=fopen (SizeInString, "w"); 
   	fprintf (outFile, "Starting Radixsort... \n ");fflush(stdout);
        fprintf (outFile, "\n %lld Milliseconds in RADIXSIMD.\n",  duration_cast<milliseconds>(stopLocal-startLocal ).count()) ;
        //cout << duration_cast<milliseconds>(stopLocal-startLocal).count() << " milliseconds in RadixSimd (cout).\n" << endl;
      	//printf ("In = %s\n", InputArr) ; //If  you want to see all the suffixes on scree
        startLocal = timer::now();    
     	for (int32_t i=0; i<sizeInArr;++i){ 
        	unsigned char *menor, *mayor;
         	//fprintf (outFile,"Suf# %i , FinalPos %d = %s\n",Suffixes[i],i, InputArr+Suffixes[i]) ; //If  you want to see all the suffixes on screen
         	//printf ("Suf# %i, FinalPos %d =    %s\n",Suffixes[i],i, InputArr+Suffixes[i]) ; //If  you want to see all the suffixes on scree
                menor=(unsigned char*)InputArr+Suffixes[i] ;  mayor=(unsigned char*)InputArr+Suffixes[i+1]; 
                a =strcmp((char *)menor, (char *)mayor);      
         	if  ((i<(sizeInArr-1))&&((*menor>*mayor)||(a>0))) {
                    fprintf(outFile,"\n ****ERROR GRAVE EN EL ORDENAMIENTO FINAL POS %i  \niMenor: %s \nMayor : %s\n ",i, menor , mayor); fflush(stdout);                
                    //printf("\n ****ERROR GRAVE EN EL ORDENAMIENTO FINAL POS %i  menor: %s\n mayor : %s\n ",i, menor , mayor); fflush(stdout);   
                }
      }
      stopLocal = timer::now();  
      fprintf ( outFile, "\n %lld Milliseconds in Veryfying Output ", duration_cast<milliseconds>(stopLocal-startLocal ).count()) ;
      if (outFile) fclose (outFile);
    #endif

    if (InputArr) free (InputArr);//Dejar solo el de arriba cuando finalicen la depuracion
    if (InputArr32) free (InputArr32);
    if (Suffixes) free (Suffixes) ;

   return 0;
}
/****************************************************END OF MAIN************************************************************/
/***************************************************************************************************************************/

void inline RadixSORT_SA_AVX512 (uint32_t SizeIn32,  uint32_t SizePadded, uint32_t *InputArr, uint32_t sizeInArr, uint32_t *SPos ){// Designed for any size input

    int32_t sizeH=SIZEHIST*sizeof(uint32_t), SizeMemPad=SizePadded*sizeof(uint32_t);  

    uint32_t *NewSortedArr; //For sorting Suffixes
    int succP_= posix_memalign ((void **) &NewSortedArr, 64, SizeMemPad);
    if (succP_!=0){         printf ("Error reserving memory for SuffixPosArray");            exit(0);    }  
    
    uint32_t *VLU_Arr;   //For Storing VLU
    succP_= posix_memalign ((void **) &VLU_Arr, 64,SizeMemPad);
    if (succP_!=0){         printf ("Error reserving memory for VLUArray");            exit(0);    }  
  
    uint32_t *VPI_Arr;   //For Storing VPI
    succP_= posix_memalign ((void **) &VPI_Arr, 64, SizeMemPad);
    if (succP_!=0){         printf ("Error reserving memory for VPIArray");            exit(0);    }  
   
    uint32_t *OutChar; //Current chars to be evaluated in each iteration
    succP_= posix_memalign ((void **) &OutChar, 64, SizeMemPad);
    if (succP_!=0){        printf ("Error reserving memory for OutChar");           exit(0);       }  

    uint32_t *Hist; //Histogram,   
    succP_= posix_memalign ((void **) &Hist, 64, sizeH);
    if (succP_!=0){        printf ("Error reserving memory for Histogram");       	exit(0);    }  

    uint32_t *Hist2; //Histogram,  For an optimization test 
    succP_= posix_memalign ((void **) &Hist2, 64, 2/*sizeH*/);
    if (succP_!=0){        printf ("Error reserving memory for Histogram");       	exit(0);    }  

    uint32_t *ExPsum; //Exclusive PrefixSum
    succP_= posix_memalign ((void **) &ExPsum, 64, sizeH);  
    if (succP_!=0){         printf ("Error reserving memory for ExPsum");              exit(0);    }

    int Rest=(sizeInArr%STEP),Diff=0;       //Because Padding  
    if (Rest>0) Diff = (STEP-Rest);         //This makes the input a mult of 16, I think that mus be done before calling Hist Function. 
    int32_t sizeInArrF= sizeInArr+Diff;   //From now on, the sizeInArrF is FIXED

    InitializeSuffixSIMD (sizeInArr, SPos); //SizeInput must be EXACT (the 8bits Inp structure)
    //InitializeSuffixSIMD (sizeInArr, NewSortedArr); //SizeInput must be EXACT (the 8bits Inp structure)
    memset(NewSortedArr,0,SizeMemPad);         //(SizePad)Este memset solo es Valido si el memcpy es Global, si es LOCAL ENTONCES USAR: //InitializeSuffixSIMD (sizeInArr, NewSortedArr); 
    //for (int i=0; i<SizePadded;++i) {      if (NewSortedArr[i]!=0)        printf ("********Initialized _NSA[%d]= %d\n  ***********",i, NewSortedArr[i]) ;             } 

    //InitializeOutPutChars(OutChar, sizeInArr);   
    memset(OutChar,0,SizeMemPad);             // Cuidado  caracter nbsp, o setear a NULL (Cero) ??? . Atencion: memset hace asignacion por BYTES asi que CUIDADO, esto no inicializa todos los elementos en OUtChar, no es como el cero. Tiene Detalle
    //for (int i=0; i<SizePadded;++i) {      if (OutChar[i]!=0)            printf ("********Initialized _OutChar[%d]= %d\n  ***********",i, OutChar[i]) ;          /* OutChar[i]=0;*/  } 


    int32_t C_RealInSize, C_Rest, C_Diff, C_InASizeFix,C_PosScatt;
    for (int32_t CurrentIt=STEP-1; CurrentIt<sizeInArrF;CurrentIt+=STEP){ // Main cicle to iterate from last char to first in text input, Starts in 1 because is when there's  more than 1 suffix to order
        
        //printf("\n START") ;fflush(stdout);
       	memset(Hist,0,sizeH);      //Como la asignacion es a cero, la dimension de la ED no genera problema.
       	//memset(Hist2,0,sizeH);      //Como la asignacion es a cero, la dimension de la ED no genera problema.
        //memset(ExPsum,0,sizeH);  // No es estrictamente necesaio puede eliminarse

        //cout << "CurrentIT Main " << CurrentIt << "\n";
        //printf("\nBefore  del SVC") ;fflush(stdout);
        SelectValidChars (InputArr, sizeInArrF,CurrentIt , SPos, OutChar, Hist2);// SIZEINPUT was exact (unfixed) , ahora is fixed para paralelismo entre chars, 
        //printf("\nDEsp del SVC") ;fflush(stdout);
       
        //Calculates the Control Variables for this iteration, 
        C_RealInSize = CurrentIt+1; 	        // Tamanyo de Entrada en la iteracion actual de acuerdo al modelo optimizado
        C_PosScatt= (sizeInArrF)-C_RealInSize  ; //Posicion donde se realizara el scatter, actualizacion de datos ordenados 
   
        /*C_Rest = (C_RealInSize%16) ;       //Rest to get to be a multiple of 16
        C_Diff = 0 ;                       //CurrDiff  //Diferencia ajustada para hacer que la entrada actual sea multipplo de 16
        if (C_Rest>0) C_Diff = (16-C_Rest); 	  //This makes the input a mult of 16, I think that mus be done before calling Hist Function.
        C_InASizeFix= C_RealInSize+C_Diff;                           //CurrentASizeFixed */

        //printf("\nPre Hist") ;fflush(stdout);
       	GetHistogram (OutChar, C_RealInSize/* C_InASizeFix*/ , Hist, VPI_Arr, VLU_Arr);	 //ESTO ES TEMPORAL PARA HACER UN HISTOGRAMA NORMAL 1 sola pasada sobre todo el Input, y por eso comento el ciclo for global uy NO ESTA SOBRE OUTCHAR		
        //printf("\nDEsp del Hist") ;fflush(stdout);

        //Hist[0]=0;//RESOLVER LUEGO DE DEPURAR SVC
      	ExcPsum (Hist, ExPsum); //Atencion: La suma del prefix sum puede sobre pasar las capacidades del uint32_t, si el tama\C3o de la entrada lo hace //+1 o Corregir manualmente Hist[0]=0
        //printf("\nDEsp del ExPsum") ;fflush(stdout);
        
     	PartialRadixSA (OutChar, C_RealInSize, ExPsum, SPos+C_PosScatt, NewSortedArr+C_PosScatt, VPI_Arr, VLU_Arr);//Sorts a complete array of  any length //Original CALL 
        //printf("\n DEsp del PartialRadix") ;fflush(stdout);

        //cout << " PosScatt " << C_PosScatt << " \n ";
        memcpy(SPos+C_PosScatt, NewSortedArr+C_PosScatt, (C_RealInSize<<2) /*x4=sizeof (uint32_t)*/ );//LAST WORKING 0310, cuidar segun el caso como se incializa NewSorted Arr OJO
        //memcpy(SPos, NewSortedArr, SizeMemPad);//Copiarlo todo No es mas rapido que las operaciones involucradas, No par atama\C3os grandes
          
      	#ifdef DEBUG_MCpy 
      	    for (int i=C_PosScatt; i<sizeInArr;++i){
            	if (NewSortedArr[i]!=SPos[i]){
             	     	printf ("\n ERROR EN COPIA VECTORIAL Pos %i SPos %i  NSA %i \n", i, SPos[i],NewSortedArr[i] ); int c=getchar() ;
	       	};  
            } 
        #endif 

        //printf("\n finish iterating \n") ;fflush(stdout);
  }//end of for CurrentIt

  if (VLU_Arr) free (VLU_Arr);
  if (VPI_Arr) free (VPI_Arr);
  if (Hist)    free(Hist); 
  if (OutChar) free (OutChar) ;
  if (ExPsum)  free (ExPsum);
  if (NewSortedArr) free (NewSortedArr) ;
}//endofRadixSort_SAi_AVX512 function



//Initialize the Suffixes array with the value SizeIn-i in the position i , this is done in the Inverse sense. Inputsize: Exact Input Size
/* Este paralelismo es factible pero amerita un estricto control sobre la cantidad de hilos generados, ya qu elos decrementos deben hacerse en relacion
a distancias diferentes, el primer hilo a 16, pero el hilo i-esimo  a i*16 (las cuales se implementarian usando desplazamientos de bits).
Excelente idea, pero a l 280918 se considera baja relacion costo beneficio. Por lo que se posterga indefinidamente,

*/

void inline InitializeSuffixSIMD (uint32_t SizeInput, uint32_t *Suffixes){  //SizeInput is measured from zero, so when calling passes SizeInput-1

    __m512i  Res, aux, Size = _mm512_set1_epi32(SizeInput);   //1. Fill  register Size  with the exact input size
   aux=Size;
   #pragma omp parallel for private(Res,aux) shared(Suffixes, SizeInput, Size)
   for (uint32_t x=0;x<SizeInput; x+=MRL  ){ 
	__m512i MASK_AVX512_prueba    = _mm512_set_epi32(x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x);
	aux= _mm512_sub_epi32 (Size, MASK_AVX512_16*MASK_AVX512_prueba);
        Res= _mm512_sub_epi32 (aux, MASK_AVX512_1to16);//2. Substract size -aux 
        _mm512_store_epi32 (Suffixes+x, Res);           //3. Store Res
        //Size= _mm512_sub_epi32 (Size, MASK_AVX512_16);  //4. Substract Size=Size-aux16
   }
   _mm512_store_epi32 (Suffixes+SizeInput, MASKZERO_AVX512/*aux*/);        //5.FakeValues for the Remaining remaining elements  

    #ifdef DEBUG_IS
    	for (int i=0; i<SizeInput;++i)         { printf ("***SuffixPos[%d]= %d\n  ***********",i, Suffixes[i]) ; }
        /* cout<< "Rest "<<Rest << "Diff" << Diff<< endl;
  	for (int i=0; i<sizeInArrF;++i) {
            if (SPos[i]!=(sizeInArr-i-1))   { 
		printf ("********Main Error Initializing SuffixPos[%d]= %d\n  ***********",i, SPos[i]) ; 
		SPos[i]=(sizeInArr-i); 
            }
         }
       int c = getchar ();*/
    #endif

}//End Initialize SuffixSIMD



//Expand unsigned char Input into uint32_t to be processed. A*256+B .-->2_8
void inline Expand_Pck2 (uint8_t *InputArr8, uint32_t *InputArr32 , uint32_t sizeInArr8){ //SizeInArr8  is the size of original input  8 bits array, no padding 

   __m128i load128;  
   __m512i load512,  Units, MuThs, Add;
   __m512i MASK_AVX512_SHIFTR_INT  =  _mm512_set_epi32(15,15,14,13,12,11,10,9, 8, 7, 6, 5,4,3,2,1);
   int step= 16-(STEP-1);
   __m512i Size = _mm512_set1_epi32(sizeInArr8);   //1. Fill  register Size  with the exact input size

   for (uint32_t i=0;i<sizeInArr8;i+=step ){ 

        load128 = _mm_loadu_si128((__m128i *)(InputArr8+i));   	//1. Sequential Load,  8 bit int 
        load512 = _mm512_cvtepu8_epi32 (load128);   		//2. Expand to 32 bits
 
        //3. Pack the input
	MuThs = _mm512_slli_epi32 (load512, 8);                                                        // Generate thousands *256->=2_8 
        Units = _mm512_mask_permutexvar_epi32(MASKZERO_AVX512, 0x7FFF,MASK_AVX512_SHIFTR_INT,load512); //Calculates Units displacing original input

	Add = _mm512_add_epi32 (MuThs, Units);                  //3.- Sum both Values

        _mm512_store_epi32 (InputArr32+i,Add);   		//4. Store in new array.
   }
   _mm512_store_epi32 (InputArr32+sizeInArr8, Size/*MASKZERO_AVX512, MASKFAKE_AVX512*/);                //Put 16 Higher al final de salida, 3 se considera el excedente *******Quizas no sea necesario.****


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

__m512i inline __attribute__((always_inline)) VPIgenerate(__m512i In){ 

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





/*inline __mmask16 VLUgenerateMASK(__m512i Conf){
   return( (__mmask16) _mm512_reduce_or_epi32(Conf) );              
}*/

//Initializes the Output Array, size must be FIXED (multiple of 16). Fills the array with invalid values
/*inline void InitializeOutPutChars(uint32_t *aux, uint32_t size) {
     _mm512_store_epi32 (aux+size, MASKFAKE_AVX512); 
     for (int32_t i=0; i<size;i+=MRL)   _mm512_store_epi32 (aux+i, MASKFAKE_AVX512); 
     
     #ifdef DEBUG_OPC
         for (int32_t i=0; i<size;++i) printf ("Output[%d]= %d\n",i, aux[i]) ; 
     #endif 
     
}*/


/*
void inline GetHistogramAVX512 (__m512i data , uint32_t *Hist, uint32_t *VPI_Arr, uint32_t *VLU_Arr, uint32_t PosVPI){

   uint32_t *auxInput;  
   __m512i outConf, VPI, VLU, VPI1, currentData, _Add;  
   __mmask16 mymask, mymask_, mymask2;

    outConf = _mm512_conflict_epi32(data);        
    VPI= VPIgenerate(outConf);
    VLU = VLUgenerate(outConf);
    VPI1 = _mm512_add_epi32(MASKONE_AVX512,VPI);   

   //  *****STORES VLU and VPI to load them when applying RadixSort************CUIDADO CON EL TEMA DE ESTAS POSICIONES ***
//    _mm512_store_epi32 ((VPI_Arr+PosVPI),VPI);
//    _mm512_store_epi32 ((VLU_Arr+PosVPI),VLU);

       //  Escenario: Vengo calculando VLU en dos operaciones: una reduccion y una compresion. Ahora lo puedo hacer todo en una reduccion y un or exclusivo, lo cual TEORICAMENTE Es mas economico.  
       // Esto me permite , al menos TEORICAMENTE otra optimizacion, ya guardar el VLU no es guardar un registro de 512 en memoria, sino guardar una mascara de 16 bits. El problema con esto es, que guardaria un 
       // VLU por iteracion, y esto no es una operacion de registros simd sino un acceso a memoria. Ademas requeriria operaciones adicionales para administrar la posicion de almacenamiento en este arreglo 
       
    mymask = _mm512_cmpeq_epu32_mask (VLU,MASKONE_AVX512);                  //Calculates the mask compressing VLU
        //mymask_= VLUgenerateMASK(outConf);//optimized
        //mymask2 = _mm512_testn_epi32_mask(outConf, MASKZERO_AVX512);
        //printf ("******** MASCARAS  ********* (deseada) mymaskCMPEQ %x vs maskReduceOr %x  Xor %x, nand %x \n", mymask, mymask_, mymask_^mask1, mymask2);
        //char c=getchar();


    currentData = _mm512_mask_i32gather_epi32 (MASKZERO_AVX512, mymask,data , (void const *)Hist,4); //bring indexes from GLOBALHist in memory
    //_mm512_i32gather_epi32 (__m512i vindex, void const* base_addr, int scale).
    //THIS OPERATION COULD BE DONE WITHOUT MASK, BUT: Discutir:  Costo de evaluacion de la mascara vs cache misses, aunque el histograma CABE en cache 
   
    _Add = _mm512_add_epi32 (currentData, VPI1);                         //Sum The local Histogram (just calculated) with the previously existing 
    _mm512_mask_i32scatter_epi32 ((void *)Hist, mymask,data , _Add, 4) ; //Update Global Hist, needs the mask for the one that has been added

	#ifdef PRINT_H         
	   //printf ("********* Histogram's Hiteration***** %u\n ", x/16);
           printf("mymask Hexa = %x \n",mymask);
           print_vector((void*)&outConf, "outCnf", "AVX512", "uint32_t", "DEC");
           print_vector((void*)&VLU,  "VLU   ", "AVX512", "uint32_t", "DEC");           
           print_vector((void*)&data, "******Data", "AVX512", "uint32_t", "DEC");
           print_vector((void*)&VPI,  "VPI   ", "AVX512", "uint32_t", "DEC");
           print_vector((void*)&VPI1,        "     VPI+1", "AVX512", "uint32_t", "DEC");   
           print_vector((void*)&currentData, "DatFromMem", "AVX512", "uint32_t", "DEC");
           print_vector((void*)&_Add,        "   SumHist", "AVX512", "uint32_t", "DEC");
           printf ("\n");
        #endif

  
   #ifdef DEBUG_H         
      int TestH[SIZEHIST];
      for (int32_t i=0; i<SIZEHIST;++i)   TestH[i]=0;   
      for (int32_t i=0; i<sizeInArr;++i)  TestH[ InputArr[i]  ]++;
      for (int32_t i=0; i<SIZEHIST;++i)   
 	    if (TestH[i]!=Hist[i] ) { printf ("CalculatedH[%d]= %d ERROR vs DesiredHist[%d]= %d \n",i, Hist[i],i,TestH[i]) ; int c=getchar();}
   #endif
   #ifdef PRINT_H         
      //for (int32_t i=0; i<SIZEINST;i+=6) printf ("Hist[%d]= %d \tHist[%d]= %d \tHist[%d]= %d\tHist[%d]= %d \t Hist[%d]= %d \t Hist[%d]= %d  \n",i, Hist[i],i+1,Hist[i+1],i+2,Hist[i+2],i+3, Hist[i+3] ,i+4, Hist[i+4],i+5, Hist[i+5] ) ;
   #endif

}*/ //end of GetHistogram







/**********OLD CODE. NOT USED ***********/
/*
//Expand unsigned char Input into uint32_t to be processed
void inline ExpandInput_old (uint8_t *InputArr8, uint32_t *InputArr32 , uint32_t sizeInArr8){ //SizeInArr8  is the size of original input  8 bits array, no padding at all

   __m512i In64_8, Thous, Hunds, Tens, Units, MuThs, MuHs, Mutns, Add1, Add2 ,Add;

   //Positions to get in every Case   
   __m512i Pos_Thous  = _mm512_set_epi8( 60,60,60,60,  56,56,56,56, 52,52,52,52, 48,48,48,48 ,44,44,44,44, 40,40,40,40, 36,36,36,36, 32,32,32,32 ,28,28,28,28, 24,24,24,24, 20,20,20,20, 16,16,16,16 ,12,12,12,12, 8,8,8,8,      4,4,4,4, 0,0,0,0 );
   __m512i Pos_Hunds  = _mm512_set_epi8( 61,61,61,61,  57,57,57,57, 53,53,53,53, 49,49,49,49 ,45,45,45,45, 41,41,41,41, 37,37,37,37, 33,33,33,33 ,29,29,29,29, 25,25,25,25, 21,21,21,21, 17,17,17,17 ,13,13,13,13, 9,9,9,9,      5,5,5,5, 1,1,1,1 );
   __m512i Pos_Tens   = _mm512_set_epi8( 62,62,62,62,  58,58,58,58, 54,54,54,54, 50,50,50,50 ,46,46,46,46, 42,42,42,42, 38,38,38,38, 34,34,34,34 ,30,30,30,30, 26,26,26,26, 22,22,22,22, 18,18,18,18 ,14,14,14,14, 10,10,10,10,  6,6,6,6, 2,2,2,2 );
   __m512i Pos_Units  = _mm512_set_epi8( 63,63,63,63,  59,59,59,59, 55,55,55,55, 51,51,51,51 ,47,47,47,47, 43,43,43,43, 39,39,39,39, 35,35,35,35 ,31,31,31,31, 27,27,27,27, 23,23,23,23, 19,19,19,19 ,15,15,15,15, 11,11,11,11,  7,7,7,7, 3,3,3,3);

   uint32_t i=0,pos512=0 ;  //inicializa la posicion de asignacion de valores en el arreglo destino a partir de la posicion 16 (dejando en 0 las 4 anteriores). O de una en 16 para asumir el padding hacia adelante y atras? 
               //Mas tiempo toma meditar en esto??? No,  el costo de implementacion o cambio es bajisimo y su impacto en el desempe\C3o es despereciable

    _mm512_store_si512 (InputArr8+sizeInArr8,  MASKZERO_AVX512);   //Put 64 Ceros al final de la ENTRADA InputArr8 , pero CUIDADO, esta dimension no la controlo yo.
    //_mm512_store_epi32 (InputArr32, MASKZERO_AVX512);              //Put 16 Ceros al principio de la SALIDA inputArr32  
    __mmask64  mask64=0x1111111111111111;  

    for (i=0;i<sizeInArr8;i+=64){  

	  In64_8 = _mm512_load_si512 (InputArr8+i); //Read 64x8bits data   

	  //2.  Aplique la permutacion 
          Thous = _mm512_maskz_shuffle_epi8 (mask64,In64_8, Pos_Thous);   //  ( 64 Elem (64x8), 512Positions: 16 Elem (16x4),)
	  Hunds = _mm512_maskz_shuffle_epi8 (mask64,In64_8, Pos_Hunds);   
	  Tens  = _mm512_maskz_shuffle_epi8 (mask64,In64_8, Pos_Tens);   
	  Units = _mm512_maskz_shuffle_epi8 (mask64,In64_8, Pos_Units);   

	  //Units = _mm512_shuffle_epi8 (In64_8, Pos_Units);
     	  //Tens  = _mm512_shuffle_epi8 (In64_8, Pos_Tens);   	  
          //Hunds = _mm512_shuffle_epi8 (In64_8, Pos_Hunds);   
          //Thous = _mm512_shuffle_epi8 (In64_8, Pos_Thous);      

	  //3 . Multiplique cada registro por el correspondiente. 
	  MuThs = _mm512_slli_epi32 (Thous, 10); //1000-->1024=2_10 
	  MuHs  = _mm512_slli_epi32 (Hunds, 7);  //100-->128= 2_7
	  Mutns = _mm512_slli_epi32 (Tens, 4);   //10-->16=2_4

	  //4. Sume Th+Hu, Te+Un. Sume ahora ambos resultados parciales, reusare registros para econcomizar
	  Add1 = _mm512_add_epi32 (Mutns, MuHs);  // Add tens and hundreds
	  Add2 = _mm512_add_epi32 (MuThs, Units); // Add Thousands and units   
	  Add  = _mm512_add_epi32 (Add1, Add2); // Add former partial results   
         
  	  //5. Almacene el producto en un arreglo de uint32_t. Este arreglo debe iniciar EN ??? A. 0 , B. 4 (previas en cero) 
          _mm512_store_epi32 (InputArr32+pos512, Add);   		

	  #ifdef DEBUG_EXPAND
      		//print_vector((void*)&In64_8,"     In64x8", "AVX512", "uint8_t", "DEC");printf("\n");

     		//print_vector((void*)&Thous,"       Thou", "AVX512", "uint32_t", "DEC");
      		//print_vector((void*)&Hunds,"       Hund", "AVX512", "uint32_t", "DEC");
      		//print_vector((void*)&Tens, "       Tens", "AVX512", "uint32_t", "DEC");
      		//print_vector((void*)&Units,"       Units", "AVX512", "uint32_t", "DEC");

     		//print_vector((void*)&MuThs,"    MuThou2", "AVX512", "uint32_t", "DEC");      		
                //print_vector((void*)&MuHs, "     MuHund", "AVX512", "uint32_t", "DEC");
      		//print_vector((void*)&Mutns,"     MuTens", "AVX512", "uint32_t", "DEC");

      		//print_vector((void*)&Add,  "   Final Add", "AVX512", "uint32_t", "DEC");
		//printf("\n");

                
      		//print_vector((void*)&Tens,    "Packed Values ", "AVX512", "uint32_t", "DEC");i
                int32_t aux8=0;
                for (uint32_t y=0;y<16;y++ ){ 
                      aux8 =  (InputArr8[(y*4)+i]*1024)+(InputArr8[(y*4)+1+i]*128)+(InputArr8[(y*4)+2+i]*16)+InputArr8[(y*4)+3+i] ;
                     //    printf(" Element[%d]: %d ",i, InputArr[i]*1024+InputArr[i+1]*128+InputArr[i+2]*16+InputArr[i+3] );
                      if (aux8 !=InputArr32[pos512+y]) {  
                           printf(" Expand Error Output32[%d]: %d  Vs %d (8 bits) \n",pos512+y, InputArr32[pos512+y], aux8 ); fflush(stdout);
		  	   int c=getchar();	
		      }
                }

	  #endif          

          pos512+=16;
    }//end for i=16

   _mm512_store_epi32 (InputArr32+pos512, MASKZERO_AVX512);              //Put 16 Ceros al final de salida 32b,  *******Quizas no sea necesario.****

  //Verifique la correctitud de las operaciones.Medite y tebga control sobre el Padding y sus efectos. Solo al final, solo al principio o ambos ? Efectos ? 

  //II. Procesar Remanente.??? . Not necessary

}//endExpandInput_old



*/


   //__m512i Pos_Thous  = _mm512_setr_epi32(0,4,8,12,16,20,24,28, 32,36,40,44,48,52,56,60);//Version reverse, could be replaced later
   //__m512i Pos_Thous  = _mm512_set_epi8( 60,56,52,48,44,40,36,32,28,24,20,16,12,8,4,0 ,60,56,52,48,44,40,36,32,28,24,20,16,12,8,4,0  ,60,56,52,48,44,40,36,32,28,24,20,16,12,8,4,0  ,60,56,52,48,44,40,36,32,28,24,20,16,12,8,4,0     );//Version reverse, could be replaced later

//   __m512i Pos_Hunds  = _mm512_setr_epi32(61,57,53,49,45,41,37,33, 29,25,21,17,13,9,5,1      1,5,9,13,17,21,25,29, 33,37,41,45,49,53,57,61);
//   __m512i Pos_Tens   = _mm512_setr_epi32(     2,6,10,14,18,22,26,30, 34,38,42,46,50,54,58,62);
//   __m512i Pos_Units  = _mm512_setr_epi32(     3,7,11,15,19,23,27,31, 35,39,43,47,51,55,59,63);


   //Masks for multiplying
   //__m512i MASK10_512   = _mm512_set_epi32(10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10);
   //__m512i MASK100_512  = _mm512_set_epi32(100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100);
   //__m512i MASK1000_512 = _mm512_set_epi32(1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000);


//This function selects the correct charts(or NULL) to be used in the next local sort.
//Los sufijos se generan de atras hacia adelante para simplificar calculos. //size a 32 , Size a 8
/*
void inline SelectValidCharSA_SIMD (uint32_t *InputText, uint32_t sizePadded,   uint32_t currentIter, uint32_t *Indexes, uint32_t SizeInput, uint32_t *OutInt){// Indexes = SortedSuffixes, SizeInput is exact

       #ifdef PRINT_SVC_SIMD
	      for (int32_t I=0; I<sizePadded+4;++I)            //printf ("Incoming Indexes [%u]  = %u ",It, Indexes[It]);
	      printf(" Val[%d] =  %d \t ", I, InputText [I] );fflush(stdout);
        
       #endif

        __m512i DispInput512, Suffix, TextIndex , Chars;
        uint32_t It, DisplaceSuf, *auxSuf, auxIt; 
        	int32_t Spadded_1= sizePadded-1;//, Spadded_1_4=Spadded_1<<2;
        //int32_t DisplInpiut=sizePadded-1-currentIter; //Este menos uno puede eliminarse si currentIter empieza en 1, pero en ese caso el control del ciculo for deberia ser <=currentIter+1, no vale la pena

        int32_t DisplInput=(sizePadded)-currentIter-1; //Este menos uno puede eliminarse si currentIter empieza en 1, pero en ese caso el control del ciculo for deberia ser <=currentIter+1, no vale la pena       
                            //DEsplazamiento horizontal, correcto. Current Iter Inicia en CERO. 

    
        //0.Partir del valor actual de current Iter. Sera hacia adelante o hacia atras?  Conviene  que sea en resta hacia atras. Pero esto repercute en Pos_SCatt y REaInSize
        //Pero si current Iter va hacia atras debo calcular el tope de como dejare de iterar , entonces calculemos aparte un Displace. En fin mucha pensadera para la poca gananacia q sera, posiblemente es lo mismo
        
        // 1. Calcule DisplaceInput. Si Current Iter va de atras hacia adelante, entonces Displace Input es igual a CurrenIter. Sino pues tocara calcularlo. Es un escalar pero  Distribuyalo a un reg AVX512.
       	DispInput512 = _mm512_set1_epi32(DisplInput);             //Horizontal Displacement value 
        DisplaceSuf= DisplInput<<2;  //3. Genere DisplaceSuf, eto es el producto de 4*DisplaceInp. Es decir shift hacia la izquierda de espaciado 2. Este es un escalar
        auxSuf=Indexes+DisplaceSuf;
        
        for (It=0;It<=currentIter;It++4){            //IT Local va de cero, 4 en 4 a Current Iter  da IGUAL pero q inicie en CEroi. 2. Este un escalar. estaba menor o igual . probar 
            auxIt= It<<2;
            //FinalDispSuf= (Spadded_1-auxIt)<<2;   
            Suffix = _mm512_load_epi32 ((void*) (auxSuf+auxIt)) ;   //   4.-  Traiga los sufijos de acuerdo a DisplaceSuf+It. Son contiguos-
       
          //   5.- Sume los valores de los sufijoos q acaba de traer con el DisplaceInput
             TextIndex= _mm512_add_epi32(Suffix, DispInput512) ;                   //3. Sum those sufixes and the Displace Value
              //Mi nueva teoria es que este valor no es necesario sumarlo, sino que cuando se recupere el valor de este sufijo .... o si , a otodos se les suma el disp horizontal ? 

          //   6.- Gather los Valores desde Inputtext usando como indices los sufijos que acaba de calcular en el paso 5.
             Chars = _mm512_i32gather_epi32 (TextIndex, (void *) InputText, 4);     //4.- Bring chars from text 

          //    7.  Almacenenlos contiguo en el Out
             
             _mm512_store_epi32 (OutInt+auxIt,Chars);                                  //5.- Store them in the OutPut

       	    #ifdef PRINT_SVC_SIMD
               cout<< " \nDisplaceInput "<< DisplInput  << " DisplaceSuf  "<< DisplaceSuf << " SizePAdded "<< sizePadded<< " CurrentIter " << currentIter<< " IT  "<< It <<"\n";  
               print_vector((void*)&Suffix,    "        Suffix", "AVX512", "uint32_t", "DEC");fflush(stdout);
               print_vector((void*)&TextIndex, "    InputIndex", "AVX512", "uint32_t", "DEC");fflush(stdout);
               print_vector((void*)&Chars,     "        Values", "AVX512", "uint32_t", "DEC");fflush(stdout);
               int c = getchar();
            #endif


        } //end for It 
        //Procesar Remanente?
        _mm512_store_epi32 ((OutInt+currentIter+4),MASKFAKE_AVX512); //1 more time, for the fakes values, En esta posicion exacta??? 
        // OutInt [auxIt] = InputText [  Indexes[ Displace+auxIt]  +  Displace ] ;   

       #ifdef PRINT_SVC_SIMD
           for (It=0;It<=currentIter;It++){            //IT Local va de cero, 4 en 4 a Current Iter  da IGUAL pero q inicie en CEroi. 2. Este un escalar. estaba menor o igual . probar 
               printf ("  CharOut[%u]=%d \t", It*4, OutInt[It*4] );        printf ("  CharOut[%u]=%d \t", (It*4)+1, OutInt[(It*4)+1] );        printf ("  CharOut[%u]=%d \t", (It*4)+2, OutInt[(It*4)+2] );       printf ("  CharOut[%u]=%d \t", (It*4)+3, OutInt[(It*4)+3] );
           }  printf ("\n");                
       #endif 
        int32_t Displace = (int32_t) (SizeInput-1)-(int32_t)currentIter;  
      
  
        auxSuf=(Indexes+Displace);        
       
	Displ = _mm512_set1_epi32(Displace);             //Displacement value 
                      
        for (It=0;It<=currentIter;It+=16){            

             //Suffix = _mm512_load_epi32 ((void*) (auxSuf+It)) ;             //2. Load contiguos suffixes
       
             TextIndex= _mm512_add_epi32(Suffix, Displ) ;                   //3. Sum those sufixes and the Displace Value

             #ifdef CHECK_Point
   	        printf ("It  %i \n", It);fflush(stdout);
                print_vector((void*)&TextIndex, "     TextIndex", "AVX512", "uint32_t", "DEC");fflush(stdout);
             #endif

             Chars = _mm512_i32gather_epi32 (TextIndex, (void *) InputText, 4);     //4.- Bring chars from text 
             _mm512_store_epi32 (OutInt+It,Chars);                                  //5.- Store them in the OutPut

       	    #ifdef PRINT_SVC_SIMD
               printf ("iIt %i  Displace %i \n", It,  Displace);fflush(stdout);
               print_vector((void*)&Suffix,     "        Suffix", "AVX512", "uint32_t", "DEC");fflush(stdout);
               print_vector((void*)&TextIndex, "     TextIndex", "AVX512", "uint32_t", "DEC");fflush(stdout);
               print_vector((void*)&Chars,     "         Chars", "AVX512", "uint32_t", "DEC");fflush(stdout);
               int c = getchar();
            #endif

        }//end of for It=0          

        _mm512_store_epi32 ((OutInt+currentIter+1),MASKFAKE_AVX512); //1 more time, for the fakes values 
        // OutInt [auxIt] = InputText [  Indexes[ Displace+auxIt]  +  Displace ] ;   
 
         #ifdef PRINT_SVC_SIMD
            printf("\nIter %u \t CurrIter %u \t SizeIn-It=  %u  \t  [Sufijo# %d ]\t TextPosition = %u  \t ", It, currentIter, SizeInput-It, Indexes[Displace+It] , Indexes[Displace+It] +Displace  ); 
            printf (" CharOut[%u]=%c \n", It,(char) OutInt[It] );
            for (int32_t i=0;i<= currentIter; ++i) printf ("OutChars[%d]=%d",i, OutInt[i]);
         #endif

}*/

//Expand unsigned char Input into uint32_t to be processed
/*
void inline Expandi_Pck4 (uint8_t *InputArr8, uint32_t *InputArr32 , uint32_t sizeInArr8){ //SizeInArr8  is the size of original input  8 bits array, no padding at all
//La otra alternativa seria con Mascaras. pudiera funcionar mucho + triple de rapido al producir 60 elementos por iteracion, el tema tambien es la suma

   __m128i load128;  
   __m512i load512,  Units, MuThs, MuHs, Mutns, Add1, Add2 ,Add, tmp;
   __m512i MASK_AVX512_SHIFTR_INT  =  _mm512_set_epi32(15,15,14,13,12,11,10,9, 8, 7, 6, 5,4,3,2,1);

   _mm512_store_si512 (InputArr8+sizeInArr8,  MASKZERO_AVX512);   //Put Ceros al final de la ENTRADA InputArr8 , pero CUIDADO, esta dimension no la controlo yo.

   for (uint32_t i=0;i<sizeInArr8;i+=13 ){ 
        load128 = _mm_loadu_si128((__m128i *)(InputArr8+i));   	//1. Sequential Load,  8 bit int 
        load512 = _mm512_cvtepu8_epi32 (load128);   		//2. Expand to 32 bits

        //3. Crea load1000, 100, 10. ( desplazar y shift load512)
	MuThs = _mm512_slli_epi32 (load512, 14); //1000-->1024=2_10 . Npo requiere ser desplazado/2_14=16384

        tmp   = _mm512_mask_permutexvar_epi32(MASKZERO_AVX512, 0x7FFF,MASK_AVX512_SHIFTR_INT,load512); //Moves to left
        MuHs  = _mm512_slli_epi32 (tmp, 7);  //100-->128= 2_7

        tmp   = _mm512_mask_permutexvar_epi32(MASKZERO_AVX512, 0x7FFF,MASK_AVX512_SHIFTR_INT,tmp); //Moves to left 
        Mutns = _mm512_slli_epi32 (tmp, 4);   //10-->16=2_4 

        Units = _mm512_mask_permutexvar_epi32(MASKZERO_AVX512, 0x7FFF,MASK_AVX512_SHIFTR_INT,tmp); //No requiere ser multiplicado

	//4. Sume Th+Hu, Te+Un. Sume ahora ambos resultados parciales, 
	Add1 = _mm512_add_epi32 (Mutns, MuHs);  // Add tens and hundreds
	Add2 = _mm512_add_epi32 (MuThs, Units); // Add Thousands and units   
	Add  = _mm512_add_epi32 (Add1, Add2);   // Add former partial results   

        _mm512_store_epi32 (InputArr32+i,Add);   		//4. Store in new array.

   }

   _mm512_store_epi32 (InputArr32+sizeInArr8, MASKZERO_AVX512);              //Put 16 Ceros al final de salida, 3 se considera el excedente *******Quizas no sea necesario.****

    #ifdef DEBUG_EXPAND
	print_vector((void*)&In64_8,"     In64x8", "AVX512", "uint8_t", "DEC");printf("\n");
	print_vector((void*)&Thous,"       Thou", "AVX512", "uint32_t", "DEC");
	print_vector((void*)&Hunds,"       Hund", "AVX512", "uint32_t", "DEC");
	print_vector((void*)&Tens, "       Tens", "AVX512", "uint32_t", "DEC");
	print_vector((void*)&Units,"       Units", "AVX512", "uint32_t", "DEC");

	//print_vector((void*)&MuThs,"    MuThou2", "AVX512", "uint32_t", "DEC");      		
        //print_vector((void*)&MuHs, "     MuHund", "AVX512", "uint32_t", "DEC");
        //print_vector((void*)&Mutns,"     MuTens", "AVX512", "uint32_t", "DEC");

      	print_vector((void*)&Add,  "   Final Add", "AVX512", "uint32_t", "DEC");
	//printf("\n");

        
   	//print_vector((void*)&Tens,    "Packed Values ", "AVX512", "uint32_t", "DEC");i
         int32_t aux8=0;
         for (uint32_t y=0;y<sizeInArr8;y++ ){ 
                      aux8 = (InputArr8[y]*256)+(InputArr8[y+1]) ;
                      //printf(" Elements[%d]: %d, %d, %d, %d \n",y, InputArr8[y],InputArr8[y+1],InputArr8[y+2],InputArr8[y+3] );
                      if (aux8 !=InputArr32[y]) {  
                           printf(" Expand Error Output32[%d]: %d  Vs %d (8 bits) \n",y, InputArr32[y], aux8 ); fflush(stdout);
		  	   //int c=getchar();	
		      }
           }
     for (uint32_t y=0;y<sizeInArr8;y++ ) printf(" Elements8[%d]: %d \n",y, InputArr8[y] );
     for (uint32_t y=0;y<sizeInArr8;y++ ) printf(" Elements32[%d]: %d \n",y, InputArr32[y] );

     #endif          
}*/ //endExpandInput
        /*#ifdef Compare2_Hist
        	for (int i=0; i<SIZEHIST;++i) {      if (Hist[i]!=Hist2[i])           printf ("********Error comparing Hist[%d]= %d vs Hist2 %d \n  ***********",i, Hist[i], Hist2[i] ) ;          } 
        #endif*/
