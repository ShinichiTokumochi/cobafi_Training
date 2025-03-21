dataset="yelp_10"
user_col="user_id"
item_col="item_id"
score_col="score"
spam_col="is_spam"


python main.py \
    --dataset $dataset \
    --user_col $user_col \
    --item_col $item_col \
    --score_col $score_col \
    --spam_col $spam_col