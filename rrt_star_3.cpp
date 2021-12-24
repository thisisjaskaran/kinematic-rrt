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

#define PI 2*acos(0.0)

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

    Node(int a = 0, int b = 0)
    {
        x = a;
        y = b;
        parent = NULL;
    }
};

struct Node* center = new Node(0,0);

double euc_distance(struct Node* node_1, struct Node* node_2)
{
    double distance = pow(pow(node_1->x - node_2->x,2) + pow(node_1->y - node_2->y,2),0.5);
    return distance;
}

struct CostFunctor
{
    struct Node* rand_node = new Node(0,0);
    struct Node* nearest_node = new Node(0,0);
    
    double gamma = 0.1;
    int height;
    double rho_local;

    long long int x1 = nearest_node->x;
    long long int y1 = height - nearest_node->y;
    long long int x2 = rand_node->x;
    long long int y2 = height - rand_node->y;
    
    double theta_1 = nearest_node->orientation;
    double theta_2 = rand_node->orientation;
    
    CostFunctor(struct Node* node_1, struct Node* node_2, int h, double r_l)
    {
        rand_node = node_1;
        nearest_node = node_2;
        height = h;
        rho_local = r_l;
    }
    
    template <typename T>
    bool operator()(const T* rho, T* residual) const
    {
        T T_x1 = T(nearest_node->x);
        T T_y1 = T(height - nearest_node->y);
        T T_x2 = T(rand_node->x);
        T T_y2 = T(height - rand_node->y);
        T T_rand_theta = T(rand_node->orientation);

        // preprocess using rho, rand_node and nearest_node to find del_x, del_y and del_theta
        T xc_alpha, yc_alpha, xc_beta, yc_beta;
        T value_given, value_obtained_alpha, value_obtained_beta;

        if(abs(tan(theta_1)) < 0.01)
        {
            xc_alpha = (T_x1 - rho[0]);
            yc_alpha = (T_y1);

            xc_beta = (T_x1 + rho[0]);
            yc_beta = (T_y1);

            value_given = T_y2 - (T_x2 - T_x1) * tan(theta_1) - T_y1;
            value_obtained_alpha = yc_alpha - (xc_alpha - T_x1) * tan(theta_1) - T_y1;
            value_obtained_beta = yc_beta - (xc_beta - T_x1) * tan(theta_1) - T_y1;
        }
        else if(abs(tan(theta_1)) > 30000)
        {
            xc_alpha = (T_x1);
            yc_alpha = (T_y1 + rho[0]);

            xc_beta = (T_x1);
            yc_beta = (T_y1 - rho[0]);

            value_given = T_y2 - (T_x2 - T_x1) * tan(theta_1) - T_y1;
            value_obtained_alpha = yc_alpha - (xc_alpha - T_x1) * tan(theta_1) - T_y1;
            value_obtained_beta = yc_beta - (xc_beta - T_x1) * tan(theta_1) - T_y1;
        }
        else
        {
            // finding center of circle
            //check once again for quadrants III and IV
            xc_alpha = (T_x1 + rho[0] * sin(theta_1));
            yc_alpha = (T_y1 - rho[0] * cos(theta_1));

            xc_beta = (T_x1 - rho[0] * sin(theta_1));
            yc_beta = (T_y1 + rho[0] * cos(theta_1));

            value_given = T_y2 - (T_x2 - T_x1) * tan(theta_1) - T_y1;
            value_obtained_alpha = yc_alpha - (xc_alpha - T_x1) * tan(theta_1) - T_y1;
            value_obtained_beta = yc_beta - (xc_beta - T_x1) * tan(theta_1) - T_y1;
        }
        T center_x, center_y;
        if(value_given > 0)
        {
            if(value_obtained_alpha > 0)
            {
                center_x = (xc_alpha);
                center_y = (yc_alpha);
            }
            else
            {
                center_x = (xc_beta);
                center_y = (yc_beta);
            }
        }
        else
        {
            if(value_obtained_alpha < 0)
            {
                center_x = (xc_alpha);
                center_y = (yc_alpha);
            }
            else
            {
                center_x = (xc_beta);
                center_y = (yc_beta);
            }
        }
        
        std::cout << "Center : " << center_x << "," << center_y << "\n";

        // finding del_x, del_y and del_theta
        T d = sqrt((T_y2 - center_y) * (T_y2 - center_y) + (T_x2 - center_x) * (T_x2 - center_x));
        T theta = atan2(T_y2 - center_y, T_x2 - center_x);

        // todo
        T inside_diff = (atan2(T_y1 - center_y, T_x1 - center_x) - atan2(T_y2 - center_y, T_x2 - center_x));

        if(inside_diff < 0)
            inside_diff *= -1;

        T del_theta = (theta_1 + inside_diff - T_rand_theta);

        if(del_theta < 0)
            del_theta *= -1;

        residual[0] = (d - rho[0]) * cos(theta);
        residual[1] = (d - rho[0]) * sin(theta);
        residual[2] = gamma * del_theta + pow(rho[0],0);
        return true;
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
    double major_axis = 0.0, minor_axis = 0.0;

    Map(int h, int w, struct Node* s, struct Node* g, double s_s, int s_r, int r_o, int n_r_o, int r_o_m_h)
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
        std::uniform_real_distribution<double> dist_height(0.0, float(height));
        std::uniform_real_distribution<double> dist_width(0.0, float(width));
        std::uniform_real_distribution<double> dist_orientation(0.0, float(4*acos(0.0)));

        struct Node* sample = new Node();
        sample->x = dist_width(mt);
        sample->y = dist_height(mt);
        sample->orientation = dist_orientation(mt) - PI;

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
    void display_arc(struct Node* node_1, struct Node* node_2, int xc, int yc, double rho)
    {
        long long int x1 = node_1->x;
        long long int y1 = node_1->y;
        long long int x2 = node_2->x;
        long long int y2 = node_2->y;
        double theta_1 = node_1->orientation;
        double theta_2 = node_2->orientation;
        
        cv::circle(world,cv::Point(round(xc),round(yc)),rho,(100,100,100),1);
    }
    // work here
    struct Node* steer(struct Node* rand_node, struct Node* nearest_node, double rho_min)
    {
        double initial_rho = rho_min;
        double rho = initial_rho;

        Problem problem;

        CostFunction* cost_function = new AutoDiffCostFunction<CostFunctor, 1, 3>(new CostFunctor(rand_node, nearest_node, height, rho));
        problem.AddResidualBlock(cost_function, nullptr, &rho);

        Solver::Options options;
        options.linear_solver_type = ceres::DENSE_QR;
        options.minimizer_progress_to_stdout = true;
        Solver::Summary summary;
        Solve(options, &problem, &summary);

        std::cout << summary.BriefReport() << "\n";
        std::cout << "rho : " << initial_rho << " -> " << rho << "\n";

        if(rho < 100.0)
            rho = 100.0;
        else if(rho > 30000.0)
            rho = 30000.0;
        
        // use obtained rho and existing rand_node and nearest_node to find steered_node
        long long int x1 = (nearest_node->x);
        long long int y1 = (height - nearest_node->y);
        long long int x2 = (rand_node->x);
        long long int y2 = (height - rand_node->y);
        double rand_theta = (rand_node->orientation);
    
        double theta_1 = nearest_node->orientation;
        double theta_2 = rand_node->orientation;

        // preprocess using rho, rand_node and nearest_node to find del_x, del_y and del_theta
        double xc_alpha, yc_alpha, xc_beta, yc_beta;
        double value_given, value_obtained_alpha, value_obtained_beta;

        // if(abs(tan(theta_1)) < 0.01)
        if(0)
        {
            xc_alpha = (x1 - rho);
            yc_alpha = (y1);

            xc_beta = (x1 + rho);
            yc_beta = (y1);

            value_given = y2 - (x2 - x1) * tan(theta_1) - y1;
            value_obtained_alpha = yc_alpha - (xc_alpha - x1) * tan(theta_1) - y1;
            value_obtained_beta = yc_beta - (xc_beta - x1) * tan(theta_1) - y1;
        }
        // else if(abs(tan(theta_1)) > 3000.0)
        else if(0)
        {
            xc_alpha = (x1);
            yc_alpha = (y1 + rho);

            xc_beta = (x1);
            yc_beta = (y1 - rho);

            value_given = y2 - (x2 - x1) * tan(theta_1) - y1;
            value_obtained_alpha = yc_alpha - (xc_alpha - x1) * tan(theta_1) - y1;
            value_obtained_beta = yc_beta - (xc_beta - x1) * tan(theta_1) - y1;
        }
        else
        {
            xc_alpha = (x1 + rho * sin(theta_1));
            yc_alpha = (y1 - rho * cos(theta_1));

            xc_beta = (x1 - rho * sin(theta_1));
            yc_beta = (y1 + rho * cos(theta_1));


            value_given = y2 - (x2 - x1) * tan(theta_1) - y1;
            value_obtained_alpha = yc_alpha - (xc_alpha - x1) * tan(theta_1) - y1;
            value_obtained_beta = yc_beta - (xc_beta - x1) * tan(theta_1) - y1;
        }

        double center_x, center_y;
        
        if(value_given > 0)
        {
            if(value_obtained_alpha > 0)
            {
                center_x = (xc_alpha);
                center_y = (yc_alpha);
            }
            else
            {
                center_x = (xc_beta);
                center_y = (yc_beta);
            }
        }
        else
        {
            if(value_obtained_alpha < 0)
            {
                center_x = (xc_alpha);
                center_y = (yc_alpha);
            }
            else
            {
                center_x = (xc_beta);
                center_y = (yc_beta);
            }
        }
    
        struct Node* center_node = new Node(round(center_x), round(height - center_y));
        double theta = atan2(rand_node->y - center_node->y, rand_node->x - center_node->x);
        
        struct Node* steered_node_1 = new Node(round(center_node->x + rho * cos(theta)), round(center_node->y - rho * sin(theta)));
        struct Node* steered_node_2 = new Node(round(center_node->x - rho * cos(theta)), round(center_node->y + rho * sin(theta)));

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

        double delta_theta = math_util::AngleDist(  atan2(nearest_node->y - center_node->y, nearest_node->x - center_node->x),
                                                    atan2(steered_node->y - center_node->y, steered_node->x - center_node->x));

        double theta_2_alpha = atan2(steered_node->x - center_node->x, steered_node->y - center_node->y);
        double theta_2_beta = math_util::AngleDiff(theta_2_alpha, PI);

        if(abs(abs(nearest_node->orientation - theta_2_alpha) - delta_theta) < 0.1)
            steered_node->orientation = theta_2_alpha;
        else
            steered_node->orientation = theta_2_beta;

        // give orientation
        std::cout << "\n" << "Nearest node : " << nearest_node->x << "," << nearest_node->y << " : " << nearest_node->orientation << "\n";
        std::cout << "Rand node : " << rand_node->x << "," << rand_node->y << " : " << rand_node->orientation << "\n";
        std::cout << "Center : " << center_node->x << "," << center_node->y << " : " << center->orientation << "\n";
        std::cout << "Theta (rand, center) : " << theta << "\n";
        std::cout << "Rho : " << rho << "\n";
        std::cout << "Steered node : " << steered_node->x << "," << steered_node->y << " : " << steered_node->orientation << "\n";
        std::cout << "del : " << euclidean_distance(rand_node,center_node) - euclidean_distance(steered_node,center_node) << "\n";
        std::cout << "New Steered node : " << steered_node->x << "," << steered_node->y << " : " << steered_node->orientation << "\n\n";

        // cv::circle(world, cv::Point(rand_node->x,rand_node->y), 2, cv::Scalar(0,0,255), cv::FILLED, cv::LINE_8);

        if((steered_node->x > 0) && (steered_node->y > 0) && (steered_node->x < width) && (steered_node->y < height))
            display_arc(nearest_node, steered_node, center_node->x, center_node->y, rho);
        else
            return nearest_node;

        return steered_node;
    }
    bool is_obstacle(struct Node* node)
    {
        if( world.at<cv::Vec3b>(node->y,node->x)[0] != 255 &&
            world.at<cv::Vec3b>(node->y,node->x)[1] != 255 &&
            world.at<cv::Vec3b>(node->y,node->x)[2] != 255)
            return true;
        return false;
    }
    // work here
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

            point_1 = cv::Point(nodes[i]->x,nodes[i]->y);
            point_2 = cv::Point(nodes[i]->x + 10 * cos(nodes[i]->orientation),nodes[i]->y - 10 * sin(nodes[i]->orientation));
            cv::arrowedLine(world, point_1, point_2, cv::Scalar(255,0,255), 2, cv::LINE_8); 
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

            // display_arc(curr_node->parent->parent, curr_node->parent);

            curr_node = curr_node->parent;
        }
        cv::Point point_1 = cv::Point(curr_node->x,curr_node->y);
        cv::Point point_2 = cv::Point(curr_node->parent->x,curr_node->parent->y);
        cv::line(world, point_1, point_2, cv::Scalar(0,0,255), 2, cv::LINE_8); 

        point_1 = cv::Point(curr_node->x,curr_node->y);
        point_2 = cv::Point(curr_node->x + 10 * cos(curr_node->orientation),curr_node->y - 10 * sin(curr_node->orientation));
        cv::arrowedLine(world, point_1, point_2, cv::Scalar(255,0,255), 2, cv::LINE_8);

        cv::Point center((start->x + goal->x) / 2, (start->y + goal->y) / 2);
        cv::Size xy(major_axis, minor_axis);
        int angle = atan2(goal->y - start->y, goal->x - start->x) * 180.0 / (2 * acos(0.0));
        int starting_point = 0;
        int ending_point = 360;
        cv::Scalar line_Color(128, 128, 128);
        int thickness = 1;

        cv::ellipse(world, center, xy, angle, starting_point, ending_point, line_Color,thickness);

        cv::imshow("Output Window",world);
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
            add_obstacle(170,170,20,20);
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
            nodes[i]->cost = nodes[i]->parent->cost + arc_distance(nodes[i],nodes[i]->parent);
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

    int randomize_obstacles = 0;
    int num_random_obstacles = 10;
    int rand_obst_min_height = 8;
    int rand_obst_max_height = 17;

    double step_size = 50.0;
    double search_radius = 40.0;
    double rho_min = 100;
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
    struct Node* goal = new Node(320,320);

    int ITERATIONS = 200000;

    Map map(height,width,start,goal,step_size,search_radius,randomize_obstacles,num_random_obstacles, rand_obst_max_height);

    map.set_obstacles(random_obstacles);

    struct Node* rand_node = new Node(start->x, start->y);
    struct Node* steered_node_global = new Node(start->x, start->y);
    struct Node* nearest_node = new Node();
    
    while(map.euclidean_distance(steered_node_global,goal) > step_size)
    {
        // sample a node
        rand_node = map.sample();

        if(rand_node->x == start->x && rand_node->y == start->y)
            continue;

        // find nearest node in existing tree
        nearest_node = map.find_nearest(rand_node);

        // needed to reject huge arcs
        double direct_distance = map.euclidean_distance(nearest_node, rand_node);

        if((direct_distance > step_size))
            continue;

        // go into gradient descent
        // using gradient descent, find arc with minimum error of known cost function
        // this node will be the steered node and should have new (x,y,theta)
        struct Node* steered_node = new Node(map.steer(rand_node,nearest_node,rho_min)->x, map.steer(rand_node,nearest_node,rho_min)->y);
        steered_node->orientation = map.steer(rand_node,nearest_node,rho_min)->orientation;

        if(steered_node->x < 0 || steered_node->y < 0)
            continue;

        std::cout << "Steering function working" << "\n";
        
        if(map.is_obstacle(steered_node))
            continue;
        
        //needs to be written correctly
        // if(map.obstacle_in_path(nearest_node,steered_node))
        //     continue;

        steered_node->parent = nearest_node;
        steered_node->cost = nearest_node->cost + map.arc_distance(nearest_node,steered_node);
        map.set_node_costs(initial_orientation);
        // steered_node->orientation = rand_node->orientation;
        
        steered_node_global = steered_node;
        
        map.nodes.push_back(steered_node);

        std::vector<struct Node*> neighbours = map.find_neighbours(steered_node);

        // map.rewire(steered_node,neighbours,rho_min, random_obstacles, initial_orientation);

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

    cv::waitKey(0);

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

        struct Node* steered_node = new Node(map.steer(rand_node,nearest_node,rho_min)->x, map.steer(rand_node,nearest_node,rho_min)->y);
    
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

        cv::waitKey(0);

        map.goal->cost = map.set_goal_cost();
    }
    cv::waitKey(0);
    cv::destroyAllWindows();

    return 0;
}