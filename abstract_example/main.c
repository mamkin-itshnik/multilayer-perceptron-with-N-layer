#include <stdlib.h>
#include <math.h>

#define LEARN_TARE 0.5
#define RAND_WEIGHT (( ((float)rand() / (float)RAND_MAX) - 0.5)* pow(pLay->out,-0.5))

//----------------------------------- NN Lay
    struct nnLay{
            //--- information about input/out width for neuro layer
           int in;
           int out;
           //--- weight matrix
           float** matrix;
           //--- current hidden value array
           float* hidden;
           //--- current errors for backPropagate
           float* errors;
    };

    struct nNetwork{
        //--- input output neuerons for NN
            int inputNeurons;
            int outputNeurons;
        //--- neuero layers count 
            int nlCount;
        //--- NN data arrays
            struct nnLay *nList;
            float *inputs;
            float *targets;
    };
        //--- neuero lay functions
        float sigmoida(float val);
        float sigmoidasDerivate(float val);
        void updMatrix(struct nnLay *pLay,float *enteredVal);
        void makeIO(struct nnLay *pLay);
        void makeHidden(struct nnLay * pLay, float *inputs);
        void calcOutError(struct nnLay * pLay,float *targets);
        void calcHidError (struct nnLay * current_lay,struct nnLay * next_lay);

         //--- neuero network functions
        void backPropagate(struct nNetwork *pNN);
        void feedForwarding(int ok, struct nNetwork *pNN);
        void train(struct nNetwork * pNN, float *in, float *targ);
        void query(struct nNetwork *pNN,  float *in);
        void printArray(float *arr,int s);

int main()
{
    printf("%s","____________hellow MPL__________ \n");
    struct nNetwork myMLP;
     //--- "Neyeral Network" this equal "NN"
    //---set width for NN input
    myMLP.inputNeurons = 100;
    //---set width for NN output
    myMLP.outputNeurons =2;
    //---set layer count for NN,
    //---where input neuerons for first layer equal NN input
    //---and output neuerons for last layer equal NN output
    myMLP.nlCount = 3;

    myMLP.nList = (struct nnLay*) malloc(( myMLP.nlCount)*sizeof(struct nnLay));

    //--- set input and output array size for every layer
    //--- where first layer have "NN input" input size
    //--- and last layer have "NN output" output size
    myMLP.nList[0].in =  myMLP.inputNeurons;
    myMLP.nList[0].out = 10;
    makeIO(&myMLP.nList[0]);

    myMLP.nList[1].in =  10;
    myMLP.nList[1].out = 4;
    makeIO(&myMLP.nList[1]);

    myMLP.nList[2].in = 4;
    myMLP.nList[2].out = myMLP.outputNeurons;
    makeIO(&myMLP.nList[2]);

    //--- create data set
    //--- (abstract data set)
        float * input_1 = (float*) malloc((100)*sizeof(float));
        for(int i = 0; i <100; i++)
        {
            input_1[i] = ((float)rand() ) / (float)RAND_MAX - 0.499f;
        }
        float * input_2 = (float*) malloc((100)*sizeof(float));
        for(int i = 0; i <100; i++)
        {
            input_2[i] = ((float)rand() ) / (float)RAND_MAX - 0.499f;
        }

    //--- create targets
        float * target_1 = (float*) malloc((2)*sizeof(float));
        target_1[0] = 0.99f;
        target_1[1] = 0.01f;

        float * target_2 = (float*) malloc((2)*sizeof(float));
        target_2[0] = 0.01f;
        target_2[1] = 0.99f;

    //--- train
           int train_calls=0;
           while(train_calls<100)
            {
                train(&myMLP,input_1,target_1);
                train(&myMLP,input_2,target_2);
                train_calls++;
            }
    //--- query
            query(&myMLP,input_1);
            query(&myMLP,input_2);
       printf("%s","_______________THE____END_______________");

    system("pause");
    return 0;
}

            float sigmoida(float val)
           {
               //--- activation function
              return (1.0 / (1.0 + exp(-val)));
           }
           float sigmoidasDerivate(float val)
           {
               //--- activation function derivative
                return (val * (1.0 - val));
           };

           void updMatrix(struct nnLay *pLay,float *enteredVal)
           {
               //--- upd weight with considering errors
               for(int ou =0; ou < pLay->out; ou++)
               {

                   for(int hid =0; hid < pLay->in; hid++)
                   {
                       pLay->matrix[hid][ou] += (LEARN_TARE * pLay->errors[ou] * enteredVal[hid]);
                   }
                   pLay->matrix[pLay->in][ou] += (LEARN_TARE * pLay->errors[ou]);
               }
           }
           void makeIO(struct nnLay *pLay)
           {
               //--- initialization values and allocating memory
               pLay->errors = (float*) malloc(( pLay->out)*sizeof(float));
                pLay->hidden = (float*) malloc(( pLay->out)*sizeof(float));

                pLay->matrix = (float**) malloc(( pLay->in+1)*sizeof(float*));
               for(int inp =0; inp <  pLay->in+1; inp++)
               {
                   pLay->matrix[inp] = (float*) malloc(pLay->out*sizeof(float));
               }
               for(int inp =0; inp < pLay->in+1; inp++)
               {
                   for(int outp =0; outp < pLay->out; outp++)
                   {
                       pLay->matrix[inp][outp] =  RAND_WEIGHT;
                   }
               }
           }

            void makeHidden(struct nnLay * pLay, float *inputs)
           {
               //--- make value after signal passing current layer
               for(int hid =0; hid < pLay->out; hid++)
               {
                   float tmpS = 0.0;
                   for(int inp =0; inp < pLay->in; inp++)
                   {
                       tmpS += inputs[inp] * pLay->matrix[inp][hid];
                   }
                   tmpS += pLay->matrix[pLay->in][hid];
                   pLay->hidden[hid] = sigmoida(tmpS);
               }
           };
           void calcOutError(struct nnLay * pLay,float *targets)
           {
               //--- calculating error if layer is last
               for(int ou =0; ou < pLay->out; ou++)
               {
                   pLay->errors[ou] = (targets[ou] - pLay->hidden[ou]) * sigmoidasDerivate(pLay->hidden[ou]);
               }
           };
           void calcHidError (struct nnLay * current_lay,struct nnLay * next_lay)
           {
               for(int hid =0; hid < next_lay->in; hid++)
               {
                   current_lay->errors[hid] = 0.0;
                   for(int ou =0; ou <  next_lay->out; ou++)
                   {
                       current_lay->errors[hid] += next_lay->errors[ou] * next_lay->matrix[hid][ou];
                   }
                   current_lay->errors[hid] *= sigmoidasDerivate(current_lay->hidden[hid]);
               }
           };

void backPropagate(struct nNetwork *pNN)
{  
    calcOutError(&(pNN->nList[pNN->nlCount -1]),pNN->targets);
    //--- for others layers to calculate errors we need information about "next layer"
    //---   //for example// to calculate 4'th layer errors we need 5'th layer errors
    for (int i = pNN->nlCount-2; i>=0; i--)
    calcHidError( &pNN->nList[i], &pNN->nList[i+1]);
    //--- updating weights
    //--- to UPD weight for current layer we must get "hidden" value array of previous layer
    for (int i = pNN->nlCount-1; i>0; i--)
    updMatrix(&pNN->nList[i],pNN->nList[i-1].hidden); 
    //--- first layer hasn't previous layer.
    //--- for him "hidden" value array of previous layer be NN input
    updMatrix(&pNN->nList[0],pNN->inputs);
}

void feedForwarding(int ok,  struct nNetwork *pNN)
{
    //--- signal through NN in forward direction
    //--- for first layer argument is _inputs
    makeHidden(&pNN->nList[0],pNN->inputs);
   //--- for other layer argument is "hidden" array previous's layer
    for (int i = 1; i<pNN->nlCount; i++)
         makeHidden(&pNN->nList[i],pNN->nList[i-1].hidden);

    //--- bool condition for query NN or train NN
    if (!ok)
    {
        printf("%s","\nFeed Forward: \n");
        for(int out =0; out < pNN->outputNeurons; out++)
        {
            printf("%.6f \n",pNN->nList[pNN->nlCount-1].hidden[out]);
        }
        return;
    }
    else
    {
        backPropagate(pNN);
    }
}

void train(struct nNetwork * pNN, float *in, float *targ)
{
    if(in && targ)
    {
    pNN->inputs = in;
    pNN->targets = targ;
      //--- bool (like 1/0)== true enable backPropogate function, else it's equal query without print
    feedForwarding( 1,pNN);
    }  
}

void query(struct nNetwork *pNN,  float *in)
{
    pNN->inputs = in;
    //--- bool (like 1/0) == false call query NN with print NN output
    feedForwarding(0,pNN);
}