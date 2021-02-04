/*Version Definitva del sort OpenMP para datos only y para Indices */

// gcc OriginalRadixSeq.c -o RadixIntOMP -fopenmp
//Original integaer radixsort OpenMP

#include<stdio.h>
#include<stdlib.h>
#include<time.h>
#include<inttypes.h>
#include<string.h>
#include <omp.h>
#define BASE_BITS 8
#define BASE (1 << BASE_BITS)
#define MASK (BASE-1)
#define DIGITS(v, shift) (((v) >> shift) & MASK)
#define DEBUG

void gen_data(uint8_t *data, uint32_t size);
void gen_dataU(unsigned *data, size_t size);
void omp_lsd_radix_sort(size_t n, unsigned data[n]);
 void gen_Indexes(uint32_t *array, uint32_t size);
void omp_lsd_radix_sortInd(size_t n, unsigned *data, uint32_t *Ind, int64_t inf, int64_t sup );

int main (int argc, char* argv[]){
//	unsigned arr1[] = {170, 45, 75, 90, 802, 24, 2, 66, 3}; //For omp     
//	int size = sizeof(arr1)/sizeof(arr1[0]);

	size_t size=(size_t) atol(argv[1]);;
	uint32_t *Indexes = (uint32_t *)calloc(size+1, sizeof(uint32_t));
	unsigned *arr1 = (unsigned *)calloc(size+1, sizeof(unsigned));
	if (arr1  == NULL) 	printf("error cadena\n");
 	printf ("size es  %lu:\n", size);

	gen_dataU(arr1, size);
	gen_Indexes(Indexes, size);
	for (int	 i = 0; i < size; i++)         	printf ("Ind: %d elem: %d ", i, arr1[i]);


	omp_lsd_radix_sortInd(size, arr1, Indexes, 0, size);
	//omp_lsd_radix_sort(size, arr1) ;
        
	for(int i=0;i<size;i++){printf("%d\n", Indexes[i]);}			

	#ifdef DEBUG
		FILE *outf = fopen("out_sorted.txt", "w");
	  	if (outf == NULL)
	    	{
	      		fprintf(stderr, "Error opening in file\n");
	      		exit(1);
	    	}
		fprintf(outf,"Tamaño %lu \n", size);
		for(size_t i=0;i<size;i++){
				fprintf(outf,"Indices [%u] %u\n ", i, Indexes[i]);
			//printf(" %d ", arr1[i]);
		}
		for(size_t i=0;i<size;i++){
				fprintf(outf,"Data pos Segun Ind[%u]=%u,  %u\n ",i,Indexes[i],  arr1[Indexes[i]]);
			//printf(" %d ", arr1[i]);		
		}		
		
		fclose(outf);
	#endif

}

void gen_data(uint8_t *data, uint32_t size){
	uint32_t i;
	uint8_t num;
	for (i=0;i<size;i++){
		num = (uint8_t)((rand()%4) + 97);//desde 65 a 90
		#ifdef DEBUG 
		printf("--%d\n", num);
		#endif
		data[i] = num;
	}
	data[size] = '\0';
	//printf("%s\n", array);		
}

void gen_dataU(unsigned *data, size_t size){
	size_t i;
	unsigned num;
	for (i=0;i<size;i++){
		num = (unsigned)((rand()%10) + 97);//desde 65 a 90
		#ifdef DEBUG_ 
		printf("--%d\n", num);
		#endif
		data[i] = num;
	}
	data[size] = '\0';
	//printf("%s\n", array);		
}


void omp_lsd_radix_sort(size_t n, unsigned data[n]) {
    unsigned * buffer = malloc(n*sizeof(unsigned));
    int total_digits = sizeof(unsigned)*8;
 
    //Each thread use local_bucket to move data
    size_t i;
    for(int shift = 0; shift < total_digits; shift+=BASE_BITS) {
        size_t bucket[BASE] = {0};
 
        size_t local_bucket[BASE] = {0}; // size needed in each bucket/thread
        //1st pass, scan whole and check the count
        #pragma omp parallel firstprivate(local_bucket)
        {
            #pragma omp for schedule(static) nowait
            for(i = 0; i < n; i++){
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
            int nthreads = omp_get_num_threads();
            int tid = omp_get_thread_num();
            for(int cur_t = nthreads - 1; cur_t >= 0; cur_t--) {
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
            for(i = 0; i < n; i++) { //note here the end condition
                buffer[local_bucket[DIGITS(data[i], shift)]++] = data[i];
            }
        }
        //now move data
        unsigned* tmp = data;
        data = buffer;
        buffer = tmp;
    }
    free(buffer);
}

void omp_lsd_radix_sortInd(size_t n, unsigned *data, uint32_t *Ind, int64_t inf, int64_t sup ) { //Cerrado , abierto
    //unsigned * buffer = (unsigned *) malloc(n*sizeof(unsigned));
    uint32_t  *outIndexes = (uint32_t  *) malloc((sup-inf)*sizeof(uint32_t ));
    int total_digits = 8; //sizeof(unsigned)*8;


    //Each thread use local_bucket to move data
    size_t i;
    for(int shift = 0; shift < total_digits; shift+=BASE_BITS) {
        size_t bucket[BASE] = {0};
 	printf("Total digits = %d\n", total_digits);
        size_t local_bucket[BASE] = {0}; // size needed in each bucket/thread
        //1st pass, scan whole and check the count
        #pragma omp parallel firstprivate(local_bucket)
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
            int nthreads = omp_get_num_threads();
            int tid = omp_get_thread_num();
            for(int cur_t = nthreads - 1; cur_t >= 0; cur_t--) {
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

// unsigned indexes[] 
void Seq_lsd_radix_sortInd(size_t n, unsigned data[], unsigned indexes[]) {
    unsigned *buffer = (unsigned *) malloc(n*sizeof(unsigned));

    int total_digits = sizeof(unsigned)*8;
    unsigned outIndexes[n]; // output array
 
    //Each thread use local_bucket to move data
    size_t i;
    for(int shift = 0; shift < total_digits; shift+=BASE_BITS) {
        size_t bucket[BASE] = {0};
 
        size_t local_bucket[BASE] = {0}; // size needed in each bucket/thread
        //1st pass, scan whole and check the count
        //#pragma omp parallel firstprivate(local_bucket)
        //{
            //#pragma omp for schedule(static) nowait
            for(i = 0; i < n; i++){
                local_bucket[DIGITS(data[i], shift)]++;
            }
            //#pragma omp critical
            for(i = 0; i < BASE; i++) {
                bucket[i] += local_bucket[i];
            }
            //#pragma omp barrier
            //#pragma omp single
            for (i = 1; i < BASE; i++) {
                bucket[i] += bucket[i - 1];
            }
            int nthreads = 1 ;//omp_get_num_threads();
            int tid = 0;  //omp_get_thread_num();
            for(int cur_t = nthreads - 1; cur_t >= 0; cur_t--) {
                if(cur_t == tid) {
                    for(i = 0; i < BASE; i++) {
                        bucket[i] -= local_bucket[i];
                        local_bucket[i] = bucket[i];
                    }
                } /*else { //just do barrier
                    #pragma omp barrier
                }*/
 
            }
          //  #pragma omp for schedule(static)
          
            for(i = 0; i < n; i++) { //note here the end condition
                long a= local_bucket[DIGITS(data[i], shift)]++;
                buffer[a/*local_bucket[DIGITS(data[i], shift)]++*/] = data[i];            /* 1. COOOOMMENTTTTT*/ //For sorting data instead of indexes
                outIndexes[a/*local_bucket[DIGITS(data[i], shift)]*/] = indexes[i];    /* INTERVENIR AQUI PARA ORDENAR INDICES**/
            }
            
            for(i = 0; i < n; i++) indexes[i]= outIndexes[i] ;                      //2. COMMENT in tests, por ser la copia de los datos
        //}
        //now move data
        unsigned* tmp = data;
        data = buffer;
        buffer = tmp;
    }
    free(buffer);
}

void gen_Indexes(uint32_t *array, uint32_t size){
	uint32_t i;
	for (i=0;i<size;i++){
		array[i] = i;
		/*#ifdef DEBUG
		printf("%d\n", i);
		#endif*/
	}
}

