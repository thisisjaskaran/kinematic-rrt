#include <iostream>
#include <cstdlib> 
#include <vector>
#include <random>
#include <math.h>
#include <map>
#include <opencv2/opencv.hpp>
#include <omp.h>
#include <cmath>

struct Node
{
    int x = 0,y = 0;
    double orientation = 0;
    double cost = 0;
    struct Node* parent;
    int flag = 0;

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
    int height, width, randomize_obstacles;
    int num_random_obstacles, rand_obst_max_height;
    double step_size, search_radius;
    struct Node* start = new Node();
    struct Node* goal = new Node();
    std::vector<struct Node*> nodes;
    cv::Mat world;

    Map(int h, int w, struct Node* s, struct Node* g, double s_s, int s_r, int r_o, int n_r_o, int r_o_m_h)
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

        randomize_obstacles = r_o;
        num_random_obstacles = n_r_o;
        rand_obst_max_height = r_o_m_h;
        
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
    double arc_distance(struct Node* node_1, struct Node* node_2)
    {
        int x1 = node_1->x;
        int y1 = node_1->y;
        int x2 = node_2->x;
        int y2 = node_2->y;
        double theta = node_1->orientation;
        
        double num = (x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1);
        double den = (2 * (x2 - x1) * sin(theta) - 2 * (y2 - y1) * cos(theta));

        double abs_den = abs(den);

        if(abs_den < 0.01)
            return euclidean_distance(node_1,node_2);

        double rho = 0.0, xc = 0.0, yc = 0.0;
        rho = num/abs_den;

        return (2 * rho * asin(euclidean_distance(node_1,node_2)/(2 * rho)));

        // consider full turn case
    }
    struct Node* sample()
    {
        std::random_device rd;
        std::mt19937 mt(rd());
        std::uniform_real_distribution<double> dist_height(0.0, float(height));
        std::uniform_real_distribution<double> dist_width(0.0, float(width));
        std::uniform_real_distribution<double> dist_orientation(0.0, float(4*acos(0.0)));

        struct Node* sample = new Node();
        sample->x = dist_width(mt);
        sample->y = dist_height(mt);
        sample->orientation = dist_orientation(mt);

        return sample;
    }
    struct Node* find_nearest(struct Node* curr_node)
    {
        double min_dist = 99999.0;
        struct Node* nearest_node;

        for(int i=0; i<nodes.size(); i++)
        {
            if(arc_distance(nodes[i],curr_node) < min_dist)
            {
                nearest_node = nodes[i];
                min_dist = arc_distance(nodes[i],curr_node);
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
        if( world.at<cv::Vec3b>(node->y,node->x)[0] != 255 &&
            world.at<cv::Vec3b>(node->y,node->x)[1] != 255 &&
            world.at<cv::Vec3b>(node->y,node->x)[2] != 255)
            return true;
        return false;
    }
    bool obstacle_in_path(struct Node* node_1, struct Node* node_2)
    {
        int resolution = int(50);

        // #pragma omp parallel for
        for(int i = 0; i <= resolution; i++)
        {
            int x_check = round((i * node_1->x + (resolution - i) * node_2->x)/resolution);
            int y_check = round((i * node_1->y + (resolution - i) * node_2->y)/resolution);

            if( world.at<cv::Vec3b>(y_check,x_check)[0] != 255 &&
                world.at<cv::Vec3b>(y_check,x_check)[1] != 255 &&
                world.at<cv::Vec3b>(y_check,x_check)[2] != 255)
                return true;
        }
        return false;
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
        
        // clearance
        int clearance = 15;
        for(int i = -clearance; i < clearance; i++)
        {
            for(int j = -clearance; j < clearance; j++)
            {
                world.at<cv::Vec3b>(start->y + j,start->x + i)[0] = 255;
                world.at<cv::Vec3b>(start->y + j,start->x + i)[1] = 255;
                world.at<cv::Vec3b>(start->y + j,start->x + i)[2] = 255;

                world.at<cv::Vec3b>(goal->y + j,goal->x + i)[0] = 255;
                world.at<cv::Vec3b>(goal->y + j,goal->x + i)[1] = 255;
                world.at<cv::Vec3b>(goal->y + j,goal->x + i)[2] = 255;
            }
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

            cv::line(world, point_1, point_2, cv::Scalar(0,255,0), 1, cv::LINE_8); 
        }
        cv::imshow("Output Window",world);
    }
    void display_final_path()
    {
        struct Node* curr_node = goal;

        while(curr_node->parent->parent != NULL)
        {
            cv::Point point_1 = cv::Point(curr_node->x,curr_node->y);
            cv::Point point_2 = cv::Point(curr_node->parent->x,curr_node->parent->y);
            cv::line(world, point_1, point_2, cv::Scalar(0,0,255), 2, cv::LINE_8); 

            point_1 = cv::Point(curr_node->x,curr_node->y);
            point_2 = cv::Point(curr_node->x + 10 * cos(curr_node->orientation),curr_node->y - 10 * sin(curr_node->orientation));
            cv::arrowedLine(world, point_1, point_2, cv::Scalar(255,0,255), 2, cv::LINE_8);

            std::cout << "(" << curr_node->x << "," << curr_node->y << ") : " << curr_node->flag << " : " << curr_node->orientation << std::endl;

            curr_node = curr_node->parent;

            // int x1 = curr_node->x;
            // int y1 = curr_node->y;
            // int x2 = curr_node->parent->x;
            // int y2 = curr_node->parent->y;
            // double theta_1 = curr_node->orientation;
            // double theta_2 = curr_node->parent->orientation;

            // double num = (x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1);
            // double den = (2 * (x2 - x1) * sin(theta_1) - 2 * (y2 - y1) * cos(theta_1));

            // double abs_den = abs(den);

            // double rho = num/abs_den;
            // double xc = 0.0, yc = 0.0;

            // if(den < 0)
            // {
            //     xc = x1 - rho * sin(theta_1);
            //     yc = y1 + rho * cos(theta_1);
            // }

            // else
            // {
            //     xc = x1 + rho * sin(theta_1);
            //     yc = y1 - rho * cos(theta_1);
            // }
            
            // cv::Point center = cv::Point(xc, yc);
            // cv::circle(world, center, rho, (255,0,0), 1, cv::LINE_8);
        }
        cv::Point point_1 = cv::Point(curr_node->x,curr_node->y);
        cv::Point point_2 = cv::Point(curr_node->parent->x,curr_node->parent->y);
        cv::line(world, point_1, point_2, cv::Scalar(0,0,255), 2, cv::LINE_8); 

        point_1 = cv::Point(curr_node->x,curr_node->y);
        point_2 = cv::Point(curr_node->x + 10 * cos(curr_node->orientation),curr_node->y - 10 * sin(curr_node->orientation));
        cv::arrowedLine(world, point_1, point_2, cv::Scalar(255,0,255), 2, cv::LINE_8);

        cv::imshow("Output Window",world);
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
            if(nodes[i] == node)
                continue;

            if(nodes[i] == node->parent)
                continue;
            
            if(obstacle_in_path(nodes[i],node))
                continue;

            if(euclidean_distance(node,nodes[i]) < search_radius)
                neighbours.push_back(nodes[i]);
        }
        return neighbours;
    }
    void set_obstacles(int random_obstacles[][4])
    {
        if(randomize_obstacles)
            set_random_obstacles(random_obstacles);
        else
        {
            // add_obstacle(50,50,50,300);
            // add_obstacle(300,50,200,50);
            // add_obstacle(100,200,50,250);
            add_obstacle(100,100,200,200);
        }
    }
    void rewire(struct Node* node, std::vector<struct Node*> neighbours, double rho_min, int random_obstacles[][4], double initial_orientation)
    {
        world = cv::Mat(height, width, CV_8UC3, cv::Scalar(255,255,255));
        set_obstacles(random_obstacles);
        for(int i = 0; i < neighbours.size(); i++)
        {
            if(obstacle_in_path(node,neighbours[i]))
                continue;
            if(!check_kino(node,neighbours[i],rho_min))
                continue;
            if(neighbours[i] == node->parent)
                continue;
            
            if(node->cost + arc_distance(node,neighbours[i]) < neighbours[i]->cost)
            {
                neighbours[i]->parent = node;
                neighbours[i]->cost = node->cost + arc_distance(node,neighbours[i]);
            }

        }

        world = cv::Mat(height, width, CV_8UC3, cv::Scalar(255,255,255));
        set_obstacles(random_obstacles);
    }
    double point_to_line(struct Node* node, struct Node* line)
    {
        int x1 = node->x;
        int y1 = node->y;
        int x2 = line->x;
        int y2 = line->y;
        double theta_2 = line->orientation;

        return abs((x1 - x2) * sin(theta_2) - (y1 - y2) * cos(theta_2));
    }
    bool check_kino(struct Node* node_1, struct Node* node_2, double rho_min)
    {
        int x1 = node_1->x;
        int y1 = height - node_1->y;
        int x2 = node_2->x;
        int y2 = height - node_2->y;
        double theta_1 = node_1->orientation;
        double theta_2 = node_2->orientation;

        double delta_theta = std::min(abs(theta_1 - theta_2), abs(4 * acos(0.0) - abs(theta_1 - theta_2)));
        double delta_tan_theta = abs(tan(theta_1) - tan(theta_2));
        
        double line_angle = 0.0;

        if(atan2(y2 - y1, x2 - x1) >= 0)
        {
            line_angle = atan2(y2 - y1, x2 - x1);
        }
        else
        {
            line_angle = 4 * acos(0.0) + atan2(y2 - y1, x2 - x1);
        }

        // going straight is okay
        if(abs(line_angle - theta_1) < 0.2)
        {
            if(abs(line_angle - theta_2) < 0.2)
                return true;
            else
                return false;
        }

        // find minimum radius of curvature
        double rho_1 = point_to_line(node_1, node_2);
        double rho_2 = point_to_line(node_2, node_1);

        double min_rho = std::min(rho_1, rho_2);

        if(min_rho < rho_min)
            return false;
        else
            return true;
    }
    double set_goal_cost()
    {
        struct Node* curr_node = goal;
        double cost = 0.0;
        while(curr_node->parent != NULL)
        {
            cost += arc_distance(curr_node,curr_node->parent);
            curr_node = curr_node->parent;
        }
        return cost;
    }
    void set_node_costs(double initial_orientation)
    {
        nodes[0]->cost = 0.0;
        nodes[0]->orientation = initial_orientation;

        for(int i = 1; i < nodes.size(); i++)
        {
            nodes[i]->cost = nodes[i]->parent->cost + arc_distance(nodes[i],nodes[i]->parent);
            // std::cout << nodes[i]->orientation << std::endl;
        }
    }
    void set_random_obstacles(int random_obstacles[][4])
    {
        for(int i = 0; random_obstacles[i][0] != '\0'; i++)
        {
            add_obstacle(random_obstacles[i][0], random_obstacles[i][1], random_obstacles[i][2], random_obstacles[i][3]);
        }
    }
};

int main()
{
    int height = 400;
    int width = 400;

    int randomize_obstacles = 1;
    int num_random_obstacles = 10;
    int rand_obst_min_height = 8;
    int rand_obst_max_height = 50;

    double step_size = 30.0;
    double search_radius = 70.0;
    double rho_min = 100;
    double initial_orientation = 7 * 3.1415 / 4;

    int random_obstacles[num_random_obstacles][4];

    for(int i = 0; i < num_random_obstacles; i++)
    {
        std::random_device rd;
        std::mt19937 mt(rd());
        std::uniform_real_distribution<double> dist_width(0.0, float(width - rand_obst_max_height - 1)); 
        std::uniform_real_distribution<double> dist_height(0.0, float(height - rand_obst_max_height - 1));
        std::uniform_real_distribution<double> dist_dim_1(8.0, float(rand_obst_max_height));
        std::uniform_real_distribution<double> dist_dim_2(8.0, float(rand_obst_max_height));

        random_obstacles[i][0] = dist_width(mt);
        random_obstacles[i][1] = dist_height(mt);
        random_obstacles[i][2] = dist_dim_1(mt);
        random_obstacles[i][3] = dist_dim_2(mt);
    }

    struct Node* start = new Node(40,40);
    start->cost = 0.0;
    start->orientation = initial_orientation;
    struct Node* goal = new Node(40,320);

    int ITERATIONS = 200000;

    Map map(height,width,start,goal,step_size,search_radius,randomize_obstacles,num_random_obstacles, rand_obst_max_height);

    map.set_obstacles(random_obstacles);

    struct Node* rand_node = new Node(start->x, start->y);
    struct Node* steered_node_global = new Node(start->x, start->y);
    struct Node* nearest_node = new Node();
    
    while(map.euclidean_distance(steered_node_global,goal) > step_size)
    {
        rand_node = map.sample();

        if(rand_node->x == start->x && rand_node->y == start->y)
            continue;

        nearest_node = map.find_nearest(rand_node);
        
        if(!map.check_kino(nearest_node,rand_node, rho_min))
            continue;

        double arc_dist = map.arc_distance(nearest_node, rand_node);

        if((arc_dist > step_size))
            continue;

        struct Node* steered_node = new Node(rand_node->x, rand_node->y);
        steered_node->orientation = rand_node->orientation;
    
        if(map.is_obstacle(steered_node))
            continue;
        
        if(map.obstacle_in_path(nearest_node,steered_node))
            continue;

        steered_node->parent = nearest_node;
        steered_node->cost = nearest_node->cost + map.arc_distance(nearest_node,steered_node);
        map.set_node_costs(initial_orientation);
        
        steered_node_global = steered_node;
        
        map.nodes.push_back(steered_node);

        std::vector<struct Node*> neighbours = map.find_neighbours(steered_node);

        map.rewire(steered_node,neighbours,rho_min, random_obstacles, initial_orientation);

        cv::circle(map.world, cv::Point(start->x,start->y), 5, cv::Scalar(0,255,0), cv::FILLED, cv::LINE_8);
        cv::circle(map.world, cv::Point(goal->x,goal->y), 5, cv::Scalar(0,0,255), cv::FILLED, cv::LINE_8);

        map.display_map();

        cv::waitKey(2);
    }

    map.goal->parent = steered_node_global;
    map.goal->cost = map.set_goal_cost();
    map.set_node_costs(initial_orientation);
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

    // #pragma omp parallel for
    for(int iter = 0; iter < ITERATIONS; iter++)
    {
        std::cout << "Iter [" << iter << "/" << ITERATIONS << "] : " << map.goal->cost << std::endl;
        
        rand_node = map.sample();

        if(rand_node->x == start->x && rand_node->y == start->y)
            continue;

        nearest_node = map.find_nearest(rand_node);

        if(!map.check_kino(nearest_node,rand_node, rho_min))
        {
            --iter;
            continue;
        }

        double arc_dist = map.arc_distance(nearest_node, rand_node);

        if(arc_dist > step_size)
        {
            --iter;
            continue;
        }

        struct Node* steered_node = new Node(rand_node->x, rand_node->y);
        steered_node->orientation = rand_node->orientation;
    
        if(map.is_obstacle(steered_node))
            continue;
        
        if(map.obstacle_in_path(nearest_node,steered_node))
            continue;

        steered_node->parent = nearest_node;
        steered_node->cost = nearest_node->cost + map.arc_distance(nearest_node,steered_node);
        map.set_node_costs(initial_orientation);
        
        steered_node_global = steered_node;
        
        map.nodes.push_back(steered_node);

        std::vector<struct Node*> neighbours = map.find_neighbours(steered_node);

        map.rewire(steered_node,neighbours,rho_min, random_obstacles, initial_orientation);

        cv::circle(map.world, cv::Point(start->x,start->y), 5, cv::Scalar(0,255,0), cv::FILLED, cv::LINE_8);
        cv::circle(map.world, cv::Point(goal->x,goal->y), 5, cv::Scalar(0,0,255), cv::FILLED, cv::LINE_8);

        map.display_map();
        map.display_final_path();

        cv::waitKey(2);

        map.goal->cost = map.set_goal_cost();
    }
    cv::waitKey(0);
    cv::destroyAllWindows();

    return 0;
}