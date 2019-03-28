from gabry_dataset_parser import get_labeled_instances
from main_tess import text_in_image


def count_clickbaits_noclickbaits_based_on(df, filter_feature):
    clickbait_df = df[(df.truthClass == 'clickbait')]
    no_clickbait_df = df[(df.truthClass == 'no-clickbait')]

    pos_clickbait = clickbait_df[clickbait_df[filter_feature] == 1]
    pos_noclickbait = no_clickbait_df[no_clickbait_df[filter_feature] == 1]

    print(f"Total: {clickbait_df.shape[0]} clickbait, {no_clickbait_df.shape[0]} no-clickbait")
    print(f"Testing {filter_feature} --> {pos_clickbait.shape[0]} positive clickbait, {pos_noclickbait.shape[0]} positive no-clickbait")
    print(f"Ratios: {pos_clickbait.shape[0] / clickbait_df.shape[0]} clickbait. {pos_noclickbait.shape[0] / no_clickbait_df.shape[0]} no-clickbait.")


labeled_instances = get_labeled_instances("./train_set/instances_converted.pickle",
                                          "./train_set/truth_converted.pickle")

labeled_instances['image_presence'] = labeled_instances.apply(lambda row: 1 if row['postMedia'] else 0, axis=1)
count_clickbaits_noclickbaits_based_on(labeled_instances, 'image_presence')


# Testing the ratios of images WITH TEXT in the obtained subset (posts with image)
clickbait_with_img = labeled_instances[
    (labeled_instances.truthClass == 'clickbait')
    &
    (labeled_instances.image_presence == 1)
]

noclickbait_with_img = labeled_instances[
    (labeled_instances.truthClass == 'no-clickbait')
    &
    (labeled_instances.image_presence == 1)
]

clickbait_with_img = clickbait_with_img.copy()
clickbait_with_img['text_in_image'] = clickbait_with_img.apply(lambda row: text_in_image(r"./train_set/" + row['postMedia'][0]), axis=1)

noclickbait_with_img = noclickbait_with_img.copy()
noclickbait_with_img['text_in_image'] = noclickbait_with_img.apply(lambda row: text_in_image(r"./train_set/" + row['postMedia'][0]), axis=1)


print()


#print(text_in_image("./train_set/media\\608674493456748545.png"))
