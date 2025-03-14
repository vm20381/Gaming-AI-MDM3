import pandas as pd
from collections import defaultdict

def extract_tags(dataset):
    all_tags = set()
    for tag_list in dataset['Tags'].dropna():
        tags = tag_list.split(', ')
        all_tags.update(tag.strip().lower() for tag in tags)
    return sorted(all_tags)

def group_by_tags(dataset, tag_groups):
    game_groups = defaultdict(set)  
    
    # Loop over the dataset to find games that match each tag group
    for _, row in dataset.iterrows():
        game_tags = set(tag.strip().lower() for tag in row['Tags'].split(', ')) if pd.notna(row['Tags']) else set()
        
        # For each group of tags defined by the user
        for group_name, group_tags in tag_groups.items():
            # If the game contains all the tags in the group, it belongs to that group
            if group_tags.issubset(game_tags):
                game_groups[group_name].add(row['Game_Name'])
    
    return game_groups 

def create_tag_groups(dataset, skip_tag_listing):
    tag_groups = {}

    if not skip_tag_listing:
        # List all available tags
        tags = extract_tags(dataset)
        print(tags)

        group_input = input("\nEnter tag combinations to group by, separated by commas and groups separated by semicolons ").strip().lower()
        
        # Split the input by semicolon to define multiple groups
        for group in group_input.split(';'):  # Split groups by semicolon
            group_tags = {tag.strip() for tag in group.split(',')}
            group_name = ', '.join(group_tags)  # Use the combination of tags as the group name
            tag_groups[group_name] = group_tags
    
    else:  # define groups without inputs
        group_input = ('fps, 3d; 2d; action, fps; multiplayer, 2d')
        for group in group_input.split(';'):  # Split groups by semicolon
            group_tags = {tag.strip() for tag in group.split(',')}
            group_name = ', '.join(group_tags)  # Use the combination of tags as the group name
            tag_groups[group_name] = group_tags
    
    return tag_groups 

def save_groups(grouped_games, filename):
    # Convert grouped games to a DataFrame
    data = []
    for group_name, games in grouped_games.items():
        for game in games:
            data.append([group_name, game])
    
    df = pd.DataFrame(data, columns=['Group', 'Game'])
    
    # Save to CSV
    df.to_csv(filename, index=False)
    print(f"\nGroups saved to {filename}")

if __name__ == "__main__":
    '''
        Run with skip_tag_listing false to select groups
        rewrite selection in group_input above and change skip_tag_listing to true to run the same groups without need for inputs
    '''
    dataset = pd.read_csv("C:/Uni/MDM3/gaming/datasets/game_data.csv")
    skip_tag_listing = True  # Set to False to list all tags 
    save = True # Set to True to save the created groups
 
    # Create the tag groups
    tag_groups = create_tag_groups(dataset, skip_tag_listing)

    # Now group the games based on the created groups
    grouped_games = group_by_tags(dataset, tag_groups)

    # Display the selected groups and grouped games
    print(f"\nSelected Groups: {tag_groups}")
    print("\nGrouped Games:")

    for group_name, games in grouped_games.items():
        print(f"\nGroup: {group_name}")
        for game in games:
            print(f" - {game}")

    if save:
        save_groups(grouped_games, "games_grouped_by_tags.csv")