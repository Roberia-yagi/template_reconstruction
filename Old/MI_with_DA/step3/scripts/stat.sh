PATTERN=$1

if [ $# != 1 ]; then
    echo "Usage: $0 PATTERN"
    exit 1
fi

RESULT_DIRS=(`ls -1 ~/nas/results/step3 | grep $PATTERN`)

echo "Model,Method,AlphaBeta,Times,Epochs,Lambda,Learning rate,Best,Mean,Median,Std,Min,Max"
for IDENTIFIER in ${RESULT_DIRS[@]}; do
    model=`echo $IDENTIFIER | cut -f2 -d'_'`
    method=`echo $IDENTIFIER | cut -f3 -d'_'`
    alpha_beta=`echo $IDENTIFIER | cut -f4 -d'_' | tr -d '\n' | xargs -0 printf "%f"`
    times=`echo $IDENTIFIER | cut -f5 -d'_' | cut -f1 -d't'`
    epochs=`echo $IDENTIFIER | cut -f6 -d'_' | cut -f1 -d'e'`
    lambda=`echo $IDENTIFIER | cut -f7 -d'_' | cut -f3 -d'a'`
    lr=`echo $IDENTIFIER | cut -f8 -d'_' | cut -f2 -d'r'`

    probs=`cat ~/nas/results/step3/$IDENTIFIER/training.log | grep "\[Stat\]" | head -n 1 | cut -f6 -d'['`
    best_prob=`echo $probs | cut -f2 -d' ' | cut -f2 -d'(' | cut -f1 -d')'`
    mean_prob=`echo $probs | cut -f3 -d' ' | cut -f2 -d'(' | cut -f1 -d')'`
    median_prob=`echo $probs | cut -f4 -d' ' | cut -f2 -d'(' | cut -f1 -d')'`
    std_prob=`echo $probs | cut -f5 -d' ' | cut -f2 -d'(' | cut -f1 -d')'`
    min_prob=`echo $probs | cut -f6 -d' ' | cut -f2 -d'(' | cut -f1 -d')'`
    max_prob=`echo $probs | cut -f7 -d' ' | cut -f2 -d'(' | cut -f1 -d')'`
    # echo $IDENTIFIER
    # echo $probs
    echo $model,$method,$alpha_beta,$times,$epochs,$lambda,$lr,$best_prob,$mean_prob,$median_prob,$std_prob,$min_prob,$max_prob
done

# model, method, epochs, lr, alpha_beta
# ./scripts/stat.sh exp1 | sort -t ',' -k 1,1 -k 2,2 -k 5,5n -k 7,7n -k 3,3n