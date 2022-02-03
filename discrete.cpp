#include <iostream>
#include <cstdlib> 
#include <vector>
#include <random>
#include <math.h>
#include <map>
#include <opencv2/opencv.hpp>
#include <omp.h>
#include <cmath>
#include <Eigen/Dense>
#include <Eigen/SVD>

#include "ceres/ceres.h"
#include "glog/logging.h"

#include "../shared/math/math_util.h"

#define PI double(2*acos(0.0))

using ceres::AutoDiffCostFunction;
using ceres::CostFunction;
using ceres::Problem;
using ceres::Solve;
using ceres::Solver;

struct Node
{
    long long int x = 0,y = 0;
    double orientation = 0;
    double cost = 0;
    struct Node* parent;
    int flag = 0;
    double rho = 0.0;
    int xc = 0, yc = 0;
    int direction = -1; // -1 : ccw, +1 : cw

    Node(int a = 0, int b = 0)
    {
        x = a;
        y = b;
        parent = NULL;
    }
};

struct Node* center = new Node(0,0);
int starting_point = 0, ending_point = 0;
long long int sample_number = -1;

double euc_distance(struct Node* node_1, struct Node* node_2)
{
    double distance = pow(pow(node_1->x - node_2->x,2) + pow(node_1->y - node_2->y,2),0.5);
    return distance;
}

class Map
{
public:
    int found = 0;
    long long int height, width, randomize_obstacles;
    int num_random_obstacles, rand_obst_max_height;
    double step_size, search_radius;
    struct Node* start = new Node();
    struct Node* goal = new Node();
    std::vector<struct Node*> nodes;
    cv::Mat world;
    double major_axis = 0.0, minor_axis = 0.0;

    Map(long long int h, long long int w, struct Node* s, struct Node* g, double s_s, int s_r, int r_o, int n_r_o, int r_o_m_h)
    {
        height = h;
        width = w;

        start->x  = s->x;
        start->y  = s->y;
        start->orientation = s->orientation;
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
    double arc_length(struct Node* nearest_node, struct Node* steered_final, int direction, double rho_final)
    {
        double theta_1 = (atan2(nearest_node->y - steered_final->yc, nearest_node->x - steered_final->xc) * 180.0) / PI;
        double theta_2 = (atan2(steered_final->y - steered_final->yc, steered_final->x - steered_final->xc) * 180.0) / PI;

        int starting_point = 0, ending_point = 0;

        if(direction == 1)
        {
            starting_point = round(theta_1);
            ending_point = round(theta_2);

            if(starting_point > ending_point)
                starting_point -= 360.0;
        }
        else
        {
            starting_point = round(theta_1);
            ending_point = round(theta_2);

            if(starting_point < ending_point)
                ending_point -= 360.0;
        }

        int delta_theta = abs(starting_point - ending_point);

        return (delta_theta * rho_final * PI / 180.0);
    }
    double euclidean_distance(struct Node* node_1, struct Node* node_2)
    {
        double distance = pow(pow(node_1->x - node_2->x,2) + pow(node_1->y - node_2->y,2),0.5);
        return distance;
    }
    double arc_distance(struct Node* node_1, struct Node* node_2)
    {
        long long int x1 = node_1->x;
        long long int y1 = node_1->y;
        long long int x2 = node_2->x;
        long long int y2 = node_2->y;
        double theta_1 = node_1->orientation;
        double theta_2 = node_2->orientation;

        double delta_theta = math_util::AngleDist(theta_1,theta_2);
        double d = euclidean_distance(node_1,node_2);

        return abs(d / (2 * sin(delta_theta/2)));
    }
    struct Node* sample()
    {
        std::random_device rd;
        std::mt19937 mt(rd());
        std::uniform_real_distribution<float> dist_height(float(0.0), float(height));
        std::uniform_real_distribution<float> dist_width(float(0.0), float(width));
        std::uniform_real_distribution<float> dist_orientation(float(0.0), float(4 * acos(0.0)));

        struct Node* sample = new Node();
        sample->x = dist_width(mt);
        sample->y = dist_height(mt);
        sample->orientation = dist_orientation(mt) - PI;

        // std::cout << "Sample x : " << sample->x << std::endl;
        // std::cout << "Sample y : " << sample->y << std::endl;
        // std::cout << "Sample orientation : " << sample->orientation << std::endl;

        if(sample_number % 5 == 0)
        {
            sample->x = goal->x;
            sample->y = goal->y;
        }

        // sample->x = 700;
        // sample->y = 300;
        // sample->orientation = -acos(0.0);
        // sample_number += 10;

        return sample;
    }
    struct Node* informed_sample(double c_best)
    {
        double c_min = euclidean_distance(start,goal);
        Eigen::MatrixXd x_centre(3,1);
        x_centre(0,0) = (start->x + goal->x) / 2;
        x_centre(1,0) = (start->y + goal->y) / 2;
        x_centre(2,0) = 0.0;

        Eigen::MatrixXd a1(3,1);
        a1(0,0) = (goal->x - start->x) / c_min;
        a1(1,0) = (goal->y - start->y) / c_min;
        a1(2,0) = 0.0;
        
        Eigen::MatrixXd i1(1,3);
        i1(0,0) = 1.0;
        i1(0,1) = 0.0;
        i1(0,2) = 0.0;
        
        Eigen::MatrixXd M = a1 * i1;

        Eigen::JacobiSVD<Eigen::MatrixXd> svd(M, Eigen::ComputeThinU | Eigen::ComputeThinV);
        
        double det_U = (svd.matrixU()).determinant();
        double det_V = (svd.matrixV()).determinant();
        int dim = M.rows();
        Eigen::MatrixXd c_middle_term(dim,dim);
        c_middle_term.setIdentity();
        c_middle_term(dim-1,dim-1) = det_U * det_V;
        Eigen::MatrixXd C = svd.matrixU() * c_middle_term * svd.matrixV().transpose();

        double diag_terms = 0.0;

        try
        {
            diag_terms = sqrt(c_best * c_best - c_min * c_min)/2;
            throw c_best;
        }
        catch (double c_best)
        {
            diag_terms = c_best/2;
        }
            
        Eigen::MatrixXd L(dim,dim);
        L = L.setIdentity() * diag_terms;
        L(0,0) = c_best / 2;

        std::random_device ball;
        std::mt19937 mt(ball());
        std::uniform_real_distribution<double> x_ball_theta_f(0.0, 1.0);
        std::uniform_real_distribution<double> x_ball_rad_f(0.0, 1.0);

        double x_ball_theta = x_ball_theta_f(mt) * 4 * acos(0.0);
        double x_ball_rad = x_ball_rad_f(mt);
        Eigen::MatrixXd x_ball(dim,1);
        x_ball(0,0) = x_ball_rad * cos(x_ball_theta);
        x_ball(1,0) = x_ball_rad * sin(x_ball_theta);
        x_ball(2,0) = 0.0;

        Eigen::MatrixXd x_f = C * L * x_ball + x_centre;
        
        struct Node* x_rand = new Node(x_f(1,0), x_f(0,0));

        // add random orientation

        major_axis = c_best;
        minor_axis = diag_terms;

        return x_rand;
    }
    struct Node* find_nearest(struct Node* curr_node)
    {
        double min_dist = 30000.0;
        struct Node* nearest_node = nodes[0];

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
    bool isValid(long long int x, long long int y)
    {
        if(x < 0)
            return false;
        else if(y < 0)
            return false;
        else if(x >= width)
            return false;
        else if(y >= height)
            return false;
        else
            return true;
    }
    void draw_arc(struct Node* nearest_node, struct Node* steered_final, int direction, double rho_final, int final=0)
    {
        cv::Point center_point(steered_final->xc,steered_final->yc);
        cv::Size xy(rho_final,rho_final);
        int angle = 0;
        int thickness = 1;

        double theta_1 = (atan2(nearest_node->y - steered_final->yc, nearest_node->x - steered_final->xc) * 180.0) / PI;
        double theta_2 = (atan2(steered_final->y - steered_final->yc, steered_final->x - steered_final->xc) * 180.0) / PI;

        int starting_point = 0, ending_point = 0;

        if(direction == 1)
        {
            starting_point = round(theta_1);
            ending_point = round(theta_2);

            if(starting_point > ending_point)
                starting_point -= 360.0;
        }
        else
        {
            starting_point = round(theta_1);
            ending_point = round(theta_2);

            if(starting_point < ending_point)
                ending_point -= 360.0;
        }

        // std::cout << "direction : " << direction << ", starting point : " << starting_point << ", ending point : " << ending_point << std::endl; 

        if(!final)
            cv::ellipse(world, center_point, xy, angle, starting_point, ending_point, (0,0,0),thickness);
        else
            cv::ellipse(world, center_point, xy, angle, starting_point, ending_point, (0,0,255),thickness);
        if(rho_final > 3000.0)
            cv::line(world, cv::Point(nearest_node->x,nearest_node->y), cv::Point(steered_final->x,steered_final->y), cv::Scalar(0,0,0), 1, cv::LINE_8);
    }
    struct Node* steer(struct Node* rand_node, struct Node* nearest_node, double rho_min, double resolution = 20.0)
    {
        long long int x1 = (nearest_node->x);
        long long int y1 = (height - nearest_node->y);
        long long int x2 = (rand_node->x);
        long long int y2 = (height - rand_node->y);
        double rand_theta = (rand_node->orientation);
    
        double theta_1 = nearest_node->orientation;
        double theta_2 = rand_node->orientation;

        double delta_min = 1e20;
        double delta_rho = 0.0;
        double delta_theta = 0.0;
        double rho_final = 0.0;
        double gamma = 10.0;

        struct Node* steered_final = new Node(0,0);
        struct Node* dummy = new Node(-1,-1);

        double xc_alpha = 0.0;
        double yc_alpha = 0.0;
        double xc_beta = 0.0;
        double yc_beta = 0.0;
        double value_given = 0.0;
        double value_obtained_alpha = 0.0;
        double value_obtained_beta = 0.0;
        double center_x, center_y;
        double error = 0.0;

        int xc_opt, yc_opt;

        for(double rho_iter = 300000.0; rho_iter >= rho_min; rho_iter -= resolution)
        {
            xc_alpha = (x1 + rho_iter * sin(theta_1));
            yc_alpha = (y1 - rho_iter * cos(theta_1));

            xc_beta = (x1 - rho_iter * sin(theta_1));
            yc_beta = (y1 + rho_iter * cos(theta_1));


            value_given = y2 - (x2 - x1) * tan(theta_1) - y1;
            value_obtained_alpha = yc_alpha - (xc_alpha - x1) * tan(theta_1) - y1;
            value_obtained_beta = yc_beta - (xc_beta - x1) * tan(theta_1) - y1;
            
            if(value_given > 0)
            {
                if(value_obtained_alpha > 0)
                {
                    center_x = xc_alpha;
                    center_y = yc_alpha;
                }
                else
                {
                    center_x = xc_beta;
                    center_y = yc_beta;
                }
            }
            else
            {
                if(value_obtained_alpha < 0)
                {
                    center_x = xc_alpha;
                    center_y = yc_alpha;
                }
                else
                {
                    center_x = xc_beta;
                    center_y = yc_beta;
                }
            }
        
            struct Node* center_node = new Node(round(center_x), round(height - center_y));
            double theta = atan2(rand_node->y - center_node->y, rand_node->x - center_node->x);
            
            struct Node* steered_node_1 = new Node(round(center_x + rho_iter * cos(theta)), round(height - center_y + rho_iter * sin(theta)));
            struct Node* steered_node_2 = new Node(round(center_x - rho_iter * cos(theta)), round(height - center_y - rho_iter * sin(theta)));

            struct Node* steered_node = new Node(0,0);

            if(euclidean_distance(steered_node_1,rand_node) < euclidean_distance(steered_node_2,rand_node))
            {
                steered_node->x = steered_node_1->x;
                steered_node->y = steered_node_1->y;
            }
            else
            {
                steered_node->x = steered_node_2->x;
                steered_node->y = steered_node_2->y;
            }

            delta_theta = math_util::AngleDiff(math_util::AngleDiff( atan2(nearest_node->y - center_node->y, nearest_node->x - center_node->x),
                                                                            atan2(steered_node->y - center_node->y, steered_node->x - center_node->x)),
                                                    0.0);

            if(delta_theta > 0)
            {
                steered_node->orientation = math_util::AngleDiff(nearest_node->orientation + abs(delta_theta), 0.0);
                steered_node->direction = 1;
            }
            else
            {
                steered_node->orientation = math_util::AngleDiff(nearest_node->orientation - abs(delta_theta), 0.0);
                steered_node->direction = -1;
            }

            steered_node->xc = center_node->x;
            steered_node->yc = center_node->y;
            steered_node->rho = rho_iter;

            delta_rho = abs(rho_iter - euclidean_distance(center_node,rand_node));
            delta_theta = abs(rand_node->orientation - steered_node->orientation);

            error = sqrt(delta_rho * delta_rho + gamma * delta_theta * delta_theta);

            if(error < delta_min)
            {
                rho_final = rho_iter;
                delta_min = error;

                steered_final->x = steered_node->x;
                steered_final->y = steered_node->y;
                steered_final->orientation = steered_node->orientation;

                steered_final->xc = center_node->x;
                steered_final->yc = center_node->y;
                steered_final->rho = rho_iter;

                if(center_node->x == round(xc_alpha)) // alpha is clockwise
                {
                    steered_final->direction = 1;
                }
                else // beta is anticlockwise
                {
                    steered_final->direction = -1;
                }
                
                xc_opt = center_node->x;
                yc_opt = center_node->y;
            }

            delete center_node;
            delete steered_node_1;
            delete steered_node_2;
            delete steered_node;
        }

        if((steered_final->x >= 0) && (steered_final->y >= 0) && (steered_final->x < width) && (steered_final->y < height))
        {
            // draw_arc(nearest_node,steered_final,steered_final->direction,rho_final);
 
            // std::cout << "----- Accepted Node Info -----" << std::endl;
            // std::cout << "Sampled node : (" << rand_node->x << " , " << rand_node->y << " , " << rand_node->orientation << ")" << std::endl;
            // std::cout << "tan(orientation) : " << tan(rand_node->orientation) << std::endl;
            // std::cout << "theta(rand,center) : " << atan2(rand_node->y - steered_final->yc, rand_node->x - steered_final->xc) * 180/PI << std::endl;
            // std::cout << "theta(steered,center) : " << atan2(steered_final->y - steered_final->yc, steered_final->x - steered_final->xc) * 180/PI << std::endl;
            // std::cout << "Nearest node : (" << nearest_node->x << " , " << nearest_node->y << " , " << nearest_node->orientation << ")" << std::endl;
            // std::cout << "Center node : (" << steered_final->xc << " , " << steered_final->yc << ")" << std::endl;
            // std::cout << "Steered node : (" << steered_final->x << " , " << steered_final->y << " , " << steered_final->orientation << ")" << std::endl;
            // std::cout << "theta_1 : " << theta_1 << ", theta_2 : " << theta_2 << std::endl;
            // std::cout << "Rho : " << rho_final << std::endl << std::endl;
            return steered_final;
        }
        else
        {
            return dummy;
        }
    }
    bool is_obstacle(struct Node* node)
    {
        if( world.at<cv::Vec3b>(node->y,node->x)[0] != 255 &&
            world.at<cv::Vec3b>(node->y,node->x)[1] != 255 &&
            world.at<cv::Vec3b>(node->y,node->x)[2] != 255)
            return true;
        return false;
        // if(world.at<cv::Vec3b>(node->x,node->y)[0] == 0 && world.at<cv::Vec3b>(node->x,node->y)[1] == 0 && world.at<cv::Vec3b>(node->x,node->y)[2] == 0)
        //     return false;
        // return true;
    }
    bool obstacle_in_path(struct Node* nearest_node, struct Node* steered_final, int direction, double rho_final)
    {
        cv::Point center_point(steered_final->xc,steered_final->yc);
        cv::Size xy(rho_final,rho_final);
        int angle = 0;
        int thickness = 1;
        
        double theta_1 = (atan2(nearest_node->y - steered_final->yc, nearest_node->x - steered_final->xc) * 180.0) / PI;
        double theta_2 = (atan2(steered_final->y - steered_final->yc, steered_final->x - steered_final->xc) * 180.0) / PI;

        int starting_point = 0, ending_point = 0;

        int test_x = 0;
        int test_y = 0;

        if(direction == 1)
        {
            starting_point = round(theta_1);
            ending_point = round(theta_2);

            if(starting_point > ending_point)
                starting_point -= 360.0;
            
            for(int theta = starting_point; theta < ending_point; theta++)
            {
                test_x = round(steered_final->xc + rho_final * cos(theta * PI / 180.0));
                test_y = round(steered_final->yc + rho_final * sin(theta * PI / 180.0));

                if(!isValid(test_x,test_y))
                    return true;

                if( world.at<cv::Vec3b>(test_y,test_x)[0] != 255 &&
                    world.at<cv::Vec3b>(test_y,test_x)[1] != 255 &&
                    world.at<cv::Vec3b>(test_y,test_x)[2] != 255)
                    return true;
            }
        }
        else
        {
            starting_point = round(theta_1);
            ending_point = round(theta_2);

            if(starting_point < ending_point)
                ending_point -= 360.0;
            
            for(int theta = starting_point; theta > ending_point; theta--)
            {
                test_x = round(steered_final->xc + rho_final * cos(theta * PI / 180.0));
                test_y = round(steered_final->yc + rho_final * sin(theta * PI / 180.0));
                
                if(!isValid(test_x,test_y))
                    return true;

                if( world.at<cv::Vec3b>(test_y,test_x)[0] != 255 &&
                    world.at<cv::Vec3b>(test_y,test_x)[1] != 255 &&
                    world.at<cv::Vec3b>(test_y,test_x)[2] != 255)
                    return true;
            }
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

            // cv::line(world, point_1, point_2, cv::Scalar(0,255,0), 1, cv::LINE_8);

            draw_arc(nodes[i]->parent, nodes[i], nodes[i]->direction, nodes[i]->rho);

            point_1 = cv::Point(nodes[i]->x,nodes[i]->y);
            point_2 = cv::Point(nodes[i]->x + 10 * cos(nodes[i]->orientation),nodes[i]->y - 10 * sin(nodes[i]->orientation));
            cv::arrowedLine(world, point_1, point_2, cv::Scalar(255,0,255), 2, cv::LINE_8); 
        }
        cv::imshow("Output Window",world);
    }
    double angle_from_center(struct Node* node_to_find, struct Node* node_center)
    {
        return (-atan2(node_center->yc - node_to_find->y, node_to_find->x - node_center->xc) * (180.0/3.141592653589793238463));
    }
    void display_final_path()
    {
        struct Node* curr_node = goal;

        while(curr_node->parent->parent != NULL)
        {
            draw_arc(curr_node->parent,curr_node,curr_node->direction,curr_node->rho,1);

            cv::Point point_1 = cv::Point(curr_node->x,curr_node->y);
            cv::Point point_2 = cv::Point(curr_node->x + 10 * cos(curr_node->orientation),curr_node->y - 10 * sin(curr_node->orientation));
            cv::arrowedLine(world, point_1, point_2, cv::Scalar(255,0,255), 2, cv::LINE_8);

            curr_node = curr_node->parent;
        }
        draw_arc(curr_node->parent,curr_node,curr_node->direction,curr_node->rho,1);

        cv::Point point_1 = cv::Point(curr_node->x,curr_node->y);
        cv::Point point_2 = cv::Point(curr_node->x + 10 * cos(curr_node->orientation),curr_node->y - 10 * sin(curr_node->orientation));
        cv::arrowedLine(world, point_1, point_2, cv::Scalar(255,0,255), 2, cv::LINE_8);

        cv::imshow("Output Window",world);
        cv::waitKey(0);
    }
    void display_nodes()
    {
        for(int i = 0; i < nodes.size(); i++)
            std::cout << i << " : " << nodes[i] << "\n";
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
            
            if(obstacle_in_path(nodes[i],node,node->direction,node->rho))
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
            add_obstacle(170,170,200,200);
        }
    }
    void rewire(struct Node* node, std::vector<struct Node*> neighbours, double rho_min, int random_obstacles[][4], double initial_orientation)
    {
        world = cv::Mat(height, width, CV_8UC3, cv::Scalar(255,255,255));
        set_obstacles(random_obstacles);
        for(int i = 0; i < neighbours.size(); i++)
        {
            if(obstacle_in_path(node,neighbours[i],neighbours[i]->direction,neighbours[i]->rho))
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
    bool check_kino(struct Node* node_1, struct Node* node_2, double rho_min)
    {
        long long int x1 = node_1->x;
        long long int y1 = height - node_1->y;
        long long int x2 = node_2->x;
        long long int y2 = height - node_2->y;
        double theta_1 = node_1->orientation;
        double theta_2 = node_2->orientation;
        
        cv::circle(world,cv::Point(x1,y1),2,(255,0,0),1);
        cv::circle(world,cv::Point(x2,y2),2,1);

        double xc_num = x2 * tan(theta_1) - x1 * tan(theta_2) - (y1 - y2) * tan(theta_1) * tan(theta_2);
        double xc_den = tan(theta_1) - tan(theta_2);

        long long int xc = round(xc_num / xc_den);
        long long int yc = round(y1 - (xc - x1) / tan(theta_1));

        double rho = sqrt((x1 - xc) * (x1 - xc) + (y1 - yc) * (y1 - yc));

        if(rho < rho_min)
            return false;
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
            nodes[i]->cost = nodes[i]->parent->cost + arc_length(nodes[i],nodes[i]->parent,nodes[i]->parent->direction,nodes[i]->parent->rho);
        }
    }
    void set_random_obstacles(int random_obstacles[][4])
    {
        for(int i = 0; random_obstacles[i][0] != '\0'; i++)
        {
            add_obstacle(random_obstacles[i][0], random_obstacles[i][1], random_obstacles[i][2], random_obstacles[i][3]);
        }
    }
    void display_node(struct Node* node, int correct = 0)
    {
        if(correct==0)
        {
            cv::Point point_1 = cv::Point(node->x,node->y);
            cv::Point point_2 = cv::Point(node->x + 10 * cos(node->orientation),node->y - 10 * sin(node->orientation));
            cv::arrowedLine(world, point_1, point_2, cv::Scalar(0,0,255), 2, cv::LINE_8);
        }
        else if(correct==1)
        {
            cv::Point point_1 = cv::Point(node->x,node->y);
            cv::Point point_2 = cv::Point(node->x + 10 * cos(node->orientation),node->y - 10 * sin(node->orientation));
            cv::arrowedLine(world, point_1, point_2, cv::Scalar(0,255,0), 2, cv::LINE_8);
        }
        else if(correct==2)
        {
            cv::Point point_1 = cv::Point(node->x,node->y);
            cv::Point point_2 = cv::Point(node->x + 10 * cos(node->orientation),node->y - 10 * sin(node->orientation));
            cv::arrowedLine(world, point_1, point_2, cv::Scalar(0,0,255), 2, cv::LINE_8);
        }
        else if(correct==10)
        {
            cv::Point point_1 = cv::Point(node->x,node->y);
            cv::Point point_2 = cv::Point(node->x + 10 * cos(node->orientation),node->y - 10 * sin(node->orientation));
            cv::arrowedLine(world, point_1, point_2, cv::Scalar(0,0,2), 2, cv::LINE_8);
        }
    }
};

int main(int argc, char **argv)
{
    int wait = 2;

    long long int height = 800;
    long long int width = 800;

    int randomize_obstacles = 1;
    int num_random_obstacles = 0;
    int rand_obst_min_height = 8;
    int rand_obst_max_height = 17;

    double step_size = 30.0;
    double search_radius = 50.0;
    double max_arc_length = 200.0;
    double distance_from_goal = 30.0;
    double rho_min = 100.0;
    double resolution = 10.0; // 300,000 to rho_min
    double initial_orientation = -1 * 3.1415 / 4;

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
    struct Node* goal = new Node(720,720);

    int ITERATIONS = 200000;

    Map map(height,width,start,goal,step_size,search_radius,randomize_obstacles,num_random_obstacles, rand_obst_max_height);

    map.set_obstacles(random_obstacles);

    struct Node* rand_node = new Node(start->x, start->y);
    struct Node* steered_node_global = new Node(start->x, start->y);
    struct Node* nearest_node = new Node();

    int count = -1;
    
    while(map.euclidean_distance(steered_node_global,goal) > distance_from_goal)
    {
        // sample a node
        rand_node = map.sample();

        if(!map.isValid(rand_node->x,rand_node->y))
        {
            // map.display_node(rand_node);
            continue;
        }

        if(rand_node->x == start->x && rand_node->y == start->y)
        {
            // map.display_node(rand_node);
            continue;
        }

        // find nearest node in existing tree
        nearest_node = map.find_nearest(rand_node);
        // nearest_node = map.start;

        // needed to reject huge arcs
        double direct_distance = map.arc_distance(nearest_node, rand_node);

        // if(direct_distance > step_size)
        // {
            // continue;
        // }

        // go into steering function
        struct Node* steered_node = new Node(0,0);
    
        steered_node->x = map.steer(rand_node,nearest_node,rho_min,resolution)->x;
        steered_node->y = map.steer(rand_node,nearest_node,rho_min,resolution)->y;
        steered_node->orientation = map.steer(rand_node,nearest_node,rho_min,resolution)->orientation;
        steered_node->xc = map.steer(rand_node,nearest_node,rho_min,resolution)->xc;
        steered_node->yc = map.steer(rand_node,nearest_node,rho_min,resolution)->yc;
        steered_node->rho = map.steer(rand_node,nearest_node,rho_min,resolution)->rho;
        steered_node->direction = map.steer(rand_node,nearest_node,rho_min,resolution)->direction;

        if(steered_node->x == -1)
        {
            // map.display_node(rand_node,2);
            continue;
        }

        if(!map.isValid(steered_node->x,steered_node->y))
        {
            // map.display_node(rand_node);
            continue;
        }

        if(map.arc_length(nearest_node,steered_node,steered_node->direction,steered_node->rho) > max_arc_length)
        {
            // std::cout << "continuing : " << map.arc_length(nearest_node,steered_node,steered_node->direction,steered_node->rho) << std::endl;
            // map.display_node(rand_node);
            continue;
        }

        // if(map.is_obstacle(steered_node))
        // {
        //     map.display_node(rand_node);
        //     continue;
        // }

        // map.display_node(rand_node,1);
        // map.display_node(steered_node,1);
        
        //needs to be written correctly
        // if(map.obstacle_in_path(nearest_node,steered_node,steered_node->direction,steered_node->rho))
        //     continue;

        steered_node->parent = nearest_node;
        steered_node->cost = nearest_node->cost + map.arc_length(nearest_node,steered_node,steered_node->direction,steered_node->rho);
        map.set_node_costs(initial_orientation);
        
        steered_node_global = steered_node;
        
        map.nodes.push_back(steered_node);

        std::vector<struct Node*> neighbours = map.find_neighbours(steered_node);

        // map.rewire(steered_node,neighbours,rho_min, random_obstacles, initial_orientation);

        cv::circle(map.world, cv::Point(start->x,start->y), 5, cv::Scalar(0,255,0), cv::FILLED, cv::LINE_8);
        cv::circle(map.world, cv::Point(goal->x,goal->y), 5, cv::Scalar(0,0,255), cv::FILLED, cv::LINE_8);

        map.display_map();

        cv::waitKey(2);
    }

    std::cout << " >>> Goal found" << std::endl;

    cv::waitKey(0);
    map.goal->parent = steered_node_global;
    map.goal->cost = map.set_goal_cost();
    map.goal->direction = steered_node_global->direction;
    map.set_node_costs(initial_orientation);
    // map.nodes.push_back(map.goal);

    cv::Point point_1 = cv::Point(steered_node_global->x,steered_node_global->y);
    cv::Point point_2 = cv::Point(nearest_node->x,nearest_node->y);

    cv::line(map.world, point_1, point_2, cv::Scalar(0,0,0), 1, cv::LINE_8); 
    
    point_1 = cv::Point(steered_node_global->x,steered_node_global->y);
    point_2 = cv::Point(goal->x,goal->y);

    cv::line(map.world, point_1, point_2, cv::Scalar(0,0,0), 1, cv::LINE_8); 

    map.display_map();
    map.display_final_path();


    // #pragma omp parallel for
    for(int iter = 0; iter < ITERATIONS; iter++)
    {
        std::cout << "Iter [" << iter << "/" << ITERATIONS << "] : " << map.goal->cost << "\n";
        
        rand_node = map.informed_sample(map.goal->cost);

        if(rand_node->x == start->x && rand_node->y == start->y)
            continue;

        nearest_node = map.find_nearest(rand_node);

        double arc_dist = map.arc_distance(nearest_node, rand_node);

        if(arc_dist > step_size)
        {
            --iter;
            continue;
        }

        struct Node* steered_node = new Node(map.steer(rand_node,nearest_node,rho_min,resolution)->x, map.steer(rand_node,nearest_node,rho_min,resolution)->y);
    
        if(map.is_obstacle(steered_node))
            continue;
        
        if(map.obstacle_in_path(nearest_node,steered_node,steered_node->direction,steered_node->rho))
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

        // map.display_map();
        map.display_final_path();

        cv::waitKey(0);

        map.goal->cost = map.set_goal_cost();
    }
    cv::waitKey(0);
    cv::destroyAllWindows();

    return 0;
}