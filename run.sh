python preprocess.py --config-path config/preprocess_t21.yaml
python generate_candidates.py --config-path config/retriever_t21.yaml
python make_features.py --config-path config/features_t21.yaml
python ranker.py --config-path config/ranker_lgb_t21.yaml
python ranker.py --config-path config/ranker_cat_t21.yaml
python ensemble.py --config-path config/ensemble.yaml