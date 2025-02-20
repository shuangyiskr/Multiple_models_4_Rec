import networkx as nx

def build_graph(train_data, user_id_map, movie_id_map):
    """
    根据训练数据构建用户-电影二部图。
    """
    graph = nx.Graph()
    
    # 添加用户节点
    for user_id in train_data["user_id"].unique():
        graph.add_node(f"user_{user_id}", node_type="user")
    
    # 添加电影节点
    for movie_id in train_data["movie_id"].unique():
        graph.add_node(f"movie_{movie_id}", node_type="movie")
    
    # 添加边（用户-电影交互）
    for _, row in train_data.iterrows():
        user_id = f"user_{row['user_id']}"
        movie_id = f"movie_{row['movie_id']}"
        graph.add_edge(user_id, movie_id, weight=row["rating"])
    
    return graph