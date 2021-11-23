#include <iostream>
#include <cstdlib> 
#include <vector>
#include <random>
#include <math.h>
#include <map>
#include <opencv2/opencv.hpp>

struct Node
{
    int x = 0,y = 0;
    double orientation = 0;
    double cost = 0;
    struct Node* parent;

    Node(int a = 0, int b = 0)
    {
        x = a;
        y = b;
        parent = NULL;
    }
};

class Map
{
public:
    int found = 0;
    int height, width;
    double step_size, search_radius;
    struct Node* start = new Node();
    struct Node* goal = new Node();
    std::vector<struct Node*> nodes;
    cv::Mat world;

    Map(int h, int w, struct Node* s, struct Node* g, double s_s, int s_r)
    {
        height = h;
        width = w;

        start->x  = s->x;
        start->y  = s->y;
        start->cost = 0;
        
        goal->x = g->x;
        goal->y = g->y;
        
        step_size = s_s;
        search_radius = s_r;
        
        world = cv::Mat(height, width, CV_8UC3, cv::Scalar(255,255,255));
        cv::circle(world, cv::Point(start->x,start->y), 5, cv::Scalar(0,255,0), cv::FILLED, cv::LINE_8);
        cv::circle(world, cv::Point(goal->x,goal->y), 5, cv::Scalar(0,0,255), cv::FILLED, cv::LINE_8);

        nodes.push_back(start);
    }
    double euclidean_distance(struct Node* node_1, struct Node* node_2)
    {
        double distance = pow(pow(node_1->x - node_2->x,2) + pow(node_1->y - node_2->y,2),0.5);
        return distance;
    }
    struct Node* sample()
    {
        std::random_device rd;
        std::mt19937 mt(rd());
        std::uniform_real_distribution<double> dist_height(0.0, float(height));
        std::uniform_real_distribution<double> dist_width(0.0, float(width));

        struct Node* sample = new Node();
        sample->x = dist_width(mt);
        sample->y = dist_height(mt);

        return sample;
    }
    struct Node* find_nearest(struct Node* curr_node)
    {
        double min_dist = 99999;
        struct Node* nearest_node;

        for(int i=0; i<nodes.size(); i++)
        {
            if(euclidean_distance(nodes[i],curr_node) < min_dist)
            {
                nearest_node = nodes[i];
                min_dist = euclidean_distance(nodes[i],curr_node);
            }
        }
        return nearest_node;
    }
    struct Node* steer(struct Node* rand_node, struct Node* nearest_node)
    {
        double distance = euclidean_distance(rand_node,nearest_node);

        if(distance < step_size)
            return rand_node;
        
        if(distance == 0)
            return nearest_node;
                
        struct Node* steered = new Node();
        steered->x = (rand_node->x - nearest_node->x) * step_size / distance + nearest_node->x;
        steered->y = (rand_node->y - nearest_node->y) * step_size / distance + nearest_node->y;

        return steered;
    }
    bool is_obstacle(struct Node* node)
    {
        if(world.at<cv::Vec3b>(node->x,node->y)[0] == 0)
            return true;
        return false;
    }
    bool obstacle_in_path(struct Node* node_1, struct Node* node_2)
    {
        if(node_1->x < node_2->x)
        {
            if(node_1->y < node_2->y)
            {
                for(int i = node_1->x + 1; i <= node_2->x; i++)
                {
                    for(int j = node_1->y + 1; j <= node_2->y; j++)
                    {
                        if(world.at<cv::Vec3b>(i,j)[0] == 0)
                            return false;
                    }
                }
            }
            else
            {
                for(int i = node_1->x + 1; i <= node_2->x; i++)
                {
                    for(int j = node_2->y + 1; j <= node_1->y; j++)
                    {
                        if(world.at<cv::Vec3b>(i,j)[0] == 0)
                            return false;
                    }
                }
            }
        }
        else
        {
            if(node_1->y < node_2->y)
            {
                for(int i = node_2->x + 1; i <= node_1->x; i++)
                {
                    for(int j = node_1->y + 1; j <= node_2->y; j++)
                    {
                        if(world.at<cv::Vec3b>(i,j)[0] == 0)
                            return false;
                    }
                }
            }
            else
            {
                for(int i = node_2->x + 1; i <= node_1->x; i++)
                {
                    for(int j = node_2->y + 1; j <= node_1->y; j++)
                    {
                        if(world.at<cv::Vec3b>(i,j)[0] == 0)
                            return false;
                    }
                }
            }
        }
        return true;
    }
    void add_obstacle(int x, int y, int height, int width)
    {
        for(int i = x; i < x + width; i++)
            for(int j = y; j < y + height; j++)
            {
                world.at<cv::Vec3b>(i,j)[0] = 0;
                world.at<cv::Vec3b>(i,j)[1] = 0;
                world.at<cv::Vec3b>(i,j)[2] = 0;
            }
    }
    void display_map(bool found = false)
    {
        for(int i = 1; i < nodes.size(); i++)
        {
            if(nodes[i]->parent == NULL)
                continue;

            cv::Point point_1 = cv::Point(nodes[i]->x,nodes[i]->y);
            cv::Point point_2 = cv::Point(nodes[i]->parent->x,nodes[i]->parent->y);

            cv::line(world, point_1, point_2, cv::Scalar(0,0,0), 1, cv::LINE_8); 
        }
        cv::imshow("img",world);
    }
    void display_final_path()
    {
        struct Node* curr_node = goal;

        while(curr_node->parent->parent != NULL)
        {
            cv::Point point_1 = cv::Point(curr_node->x,curr_node->y);
            cv::Point point_2 = cv::Point(curr_node->parent->x,curr_node->parent->y);
            cv::line(world, point_1, point_2, cv::Scalar(0,0,255), 2, cv::LINE_8); 

            curr_node = curr_node->parent;
        }
        cv::Point point_1 = cv::Point(curr_node->x,curr_node->y);
        cv::Point point_2 = cv::Point(curr_node->parent->x,curr_node->parent->y);
        cv::line(world, point_1, point_2, cv::Scalar(0,0,255), 2, cv::LINE_8); 

        cv::imshow("img",world);
    }
    void display_nodes()
    {
        for(int i = 0; i < nodes.size(); i++)
            std::cout << i << " : " << nodes[i] << std::endl;
    }
    std::vector<struct Node*> find_neighbours(struct Node* node)
    {
        std::vector<struct Node*> neighbours;
        for(int i = 0; i < nodes.size(); i++)
        {
            if(euclidean_distance(node,nodes[i]) < search_radius)
                neighbours.push_back(nodes[i]);
        }
        return neighbours;
    }
    void rewire(struct Node* node, std::vector<struct Node*> neighbours)
    {
        world = cv::Mat(height, width, CV_8UC3, cv::Scalar(255,255,255));
        for(int i = 0; i < neighbours.size(); i++)
        {
            if(neighbours[i] == node->parent)
                continue;
            
            if(node->cost + euclidean_distance(node,neighbours[i]) < neighbours[i]->cost)
            {
                neighbours[i]->parent = node;
                neighbours[i]->cost = node->cost + euclidean_distance(node,neighbours[i]);
            }
        }
    }
};

int main()
{
    int height = 400;
    int width = 600;

    struct Node* start = new Node(40,40);
    start->cost = 0;
    struct Node* goal = new Node(350,350);

    double step_size =15.0, search_radius = 19.0 ;
    int ITERATIONS = 200000;

    Map map(height,width,start,goal,step_size,search_radius);
    map.add_obstacle(100,100,200,200);

    struct Node* rand_node = new Node(start->x, start->y);
    struct Node* steered_node_global = new Node(start->x, start->y);
    struct Node* nearest_node = new Node();
    
    while(map.euclidean_distance(steered_node_global,goal) > step_size)
    {
        rand_node = map.sample();

        if(rand_node->x == start->x && rand_node->y == start->y)
            continue;

        nearest_node = map.find_nearest(rand_node);

        struct Node* steered_node = new Node(map.steer(rand_node,nearest_node)->x, map.steer(rand_node,nearest_node)->y);
        steered_node->parent = nearest_node;
        steered_node->cost = nearest_node->cost + map.euclidean_distance(nearest_node,steered_node);
        
        steered_node_global = steered_node;
    
        if(map.is_obstacle(steered_node))
            continue;
        
        if(map.obstacle_in_path(nearest_node,steered_node))
            continue;
        
        map.nodes.push_back(steered_node);

        std::vector<struct Node*> neighbours = map.find_neighbours(steered_node);

        map.rewire(steered_node,neighbours);

        map.world = cv::Mat(height, width, CV_8UC3, cv::Scalar(255,255,255));
        map.add_obstacle(100,100,200,200);
        cv::circle(map.world, cv::Point(start->x,start->y), 5, cv::Scalar(0,255,0), cv::FILLED, cv::LINE_8);
        cv::circle(map.world, cv::Point(goal->x,goal->y), 5, cv::Scalar(0,0,255), cv::FILLED, cv::LINE_8);

        map.display_map();

        cv::waitKey(2);
    }

    map.goal->parent = steered_node_global;
    map.goal->cost = steered_node_global->cost + map.euclidean_distance(steered_node_global,map.goal);
    map.nodes.push_back(map.goal);

    cv::Point point_1 = cv::Point(steered_node_global->x,steered_node_global->y);
    cv::Point point_2 = cv::Point(nearest_node->x,nearest_node->y);

    cv::line(map.world, point_1, point_2, cv::Scalar(0,0,0), 1, cv::LINE_8); 
    
    point_1 = cv::Point(steered_node_global->x,steered_node_global->y);
    point_2 = cv::Point(goal->x,goal->y);

    cv::line(map.world, point_1, point_2, cv::Scalar(0,0,0), 1, cv::LINE_8); 

    map.display_map();
    map.display_final_path();

    // cv::waitKey(0);

    for(int iter = 0; iter < ITERATIONS; iter++)
    {
        std::cout << "Iter [" << iter << "/" << ITERATIONS << "]" << std::endl;
        rand_node = map.sample();

        if(rand_node->x == start->x && rand_node->y == start->y)
            continue;

        nearest_node = map.find_nearest(rand_node);

        struct Node* steered_node = new Node(map.steer(rand_node,nearest_node)->x, map.steer(rand_node,nearest_node)->y);
        steered_node->parent = nearest_node;
        
        steered_node_global = steered_node;
    
        if(map.is_obstacle(steered_node))
            continue;
        
        if(map.obstacle_in_path(nearest_node,steered_node))
            continue;
        
        steered_node->cost = nearest_node->cost + map.euclidean_distance(nearest_node,steered_node);
        map.nodes.push_back(steered_node);

        std::vector<struct Node*> neighbours = map.find_neighbours(steered_node);

        map.rewire(steered_node,neighbours);

        map.world = cv::Mat(height, width, CV_8UC3, cv::Scalar(255,255,255));
        map.add_obstacle(100,100,200,200);
        cv::circle(map.world, cv::Point(start->x,start->y), 5, cv::Scalar(0,255,0), cv::FILLED, cv::LINE_8);
        cv::circle(map.world, cv::Point(goal->x,goal->y), 5, cv::Scalar(0,0,255), cv::FILLED, cv::LINE_8);

        map.display_map();
        map.display_final_path();

        cv::waitKey(2);
    }
    cv::waitKey(0);
    cv::destroyAllWindows();

    return 0;
}