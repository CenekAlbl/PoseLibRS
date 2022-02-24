#include "r6p.h"
#include "PoseLib/misc/utils.h"
#include "PoseLib/misc/quaternion.h"

namespace poselib{

struct R6PIterResult{
    Eigen::Vector3d w;
    Eigen::Vector3d C;
    Eigen::Vector3d v;
    Eigen::Vector3d t;
};

double calcErrAlgebraicR6PFocalRadialDoubleLin(Eigen::Vector3d vr, Eigen::Vector3d Cr, Eigen::Vector3d wr, Eigen::Vector3d tr, Eigen::MatrixXd X, Eigen::MatrixXd u){
    double err = 0;
    for (int i = 0; i < X.cols(); i++)
    {
        Eigen::Vector3d uh; 
        Eigen::Matrix3d I = Eigen::Matrix3d::Identity();
        double rs = u(1,i);
        Eigen::Vector3d eq = X_(u.col(i))*((I + rs*X_(wr))*(I + X_(vr))*X.col(i) + Cr + rs*tr);
        err += eq.cwiseAbs().sum();
    }
    return err;
}

void inputSwitchDirection(Eigen::MatrixXd &X, Eigen::MatrixXd &u){
	Eigen::VectorXd temp1 = X.row(0);
	Eigen::VectorXd temp2 = u.row(0);
	X.row(0) = X.row(1);
	X.row(1) = temp1;
	u.row(0) = u.row(1);
	u.row(1) = temp2; 
}


RSCameraPose outputSwitchDirection(const RSCameraPose & in){
	RSCameraPose out;
    Eigen::Matrix3d R = in.R();
    Eigen::Vector3d temp;
    temp = R.row(1);
    R.row(1) = R.row(0);
    R.row(0) = temp;
    out.q = rotmat_to_quat(R);
	out.w(0) = - in.w(1);
	out.w(1) = - in.w(0);
	out.w(2) = - in.w(2);
	out.t(0) = in.t(1);
	out.t(1) = in.t(0);
	out.t(2) = in.t(2);
	out.v(0) = in.v(1);
	out.v(1) = in.v(0);
	out.v(2) = in.v(2);	

	return out;
}

R6PIterResult R6PIter(const Eigen::MatrixXd & X, const Eigen::MatrixXd & u, const Eigen::Vector3d & vk){
    Eigen::Matrix<double,12,13> A = Eigen::Matrix<double,12,13>::Zero();
    A.col(0).head(6) << X.row(2).transpose().array() + X.row(1).transpose().array() * u.row(1).transpose().array();  
    A.col(0).tail(6) << -X.row(1).transpose().array() * u.row(0).transpose().array();
    A.col(1).head(6) << -X.row(0).transpose().array() * u.row(1).transpose().array();
    A.col(1).tail(6) << X.row(2).transpose().array() + X.row(0).transpose().array() * u.row(0).transpose().array();
    A.col(2).head(6) << - X.row(0).transpose().array();
    A.col(2).tail(6) << - X.row(1).transpose().array();
    A.col(3).head(6) << X.row(0).transpose().array() * vk(1) * (-u.row(0).transpose().array()) - X.row(2).transpose().array() * (- u.row(0).transpose().array()) - u.row(1).transpose().array() * (X.row(1).transpose().array() * ( -u.row(0).transpose().array()) + X.row(0).transpose().array() * vk(2) * (-u.row(0).transpose().array()) - X.row(2).transpose().array() * vk(0) * (-u.row(0).transpose().array())) - X.row(1).transpose().array() * vk(0) * (-u.row(0).transpose().array());
    A.col(3).tail(6) << u.row(0).transpose().array() * (X.row(1).transpose().array() * (-u.row(0).transpose().array()) + X.row(0).transpose().array() * vk(2) * (-u.row(0).transpose().array()) - X.row(2).transpose().array() * vk(0) * (-u.row(0).transpose().array()));
    A.col(4).head(6) << u.row(1).transpose().array() * (X.row(0).transpose().array() * (-u.row(0).transpose().array()) - X.row(1).transpose().array() * vk(2) * (-u.row(0).transpose().array()) + X.row(2).transpose().array() * vk(1) * (-u.row(0).transpose().array()));
    A.col(4).tail(6) << X.row(0).transpose().array() * vk(1) * (-u.row(0).transpose().array()) - X.row(2).transpose().array() * (-u.row(0).transpose().array()) - u.row(0).transpose().array() * (X.row(0).transpose().array() * (-u.row(0).transpose().array()) -X.row(1).transpose().array() * vk(2) * (-u.row(0).transpose().array()) + X.row(2).transpose().array() * vk(1) * (-u.row(0).transpose().array())) - X.row(1).transpose().array() * vk(0) * (-u.row(0).transpose().array());
    A.col(5).head(6) << X.row(0).transpose().array() * (-u.row(0).transpose().array()) - X.row(1).transpose().array() * vk(2) * (-u.row(0).transpose().array()) + X.row(2).transpose().array() * vk(1) * (-u.row(0).transpose().array());
    A.col(5).tail(6) << X.row(1).transpose().array() * (-u.row(0).transpose().array()) + X.row(0).transpose().array() * vk(2) * (-u.row(0).transpose().array()) - X.row(2).transpose().array() * vk(0) * (-u.row(0).transpose().array());
    A.col(6).tail(6) << Eigen::MatrixXd::Ones(6,1);
    A.col(7).head(6) << -Eigen::MatrixXd::Ones(6,1);
    A.col(8).head(6) << u.row(1).transpose().array();
    A.col(8).tail(6) << -u.row(0).transpose().array();
    A.col(9).tail(6) << u.row(0).transpose().array();
    A.col(10).head(6) << -u.row(0).transpose().array();
    A.col(11).head(6) << -u.row(1).transpose().array() * (-u.row(0).transpose().array());
    A.col(11).tail(6) << u.row(0).transpose().array() * (-u.row(0).transpose().array());
    A.col(12).head(6) << X.row(2).transpose().array() * u.row(1).transpose().array() - X.row(1).transpose().array();
    A.col(12).tail(6) << X.row(0).transpose().array() - X.row(2).transpose().array() * u.row(0).transpose().array();
    Eigen::MatrixXd n = A.fullPivLu().kernel();

    int end = n.cols()-1;
    double s = n(12,end);

    Eigen::Vector3d v = n.col(end).head(3)/s; 
    Eigen::Vector3d w = n.col(end).segment(3,3)/s;
    Eigen::Vector3d C = n.col(end).segment(6,3)/s;
    Eigen::Vector3d t = n.col(end).segment(9,3)/s;


    return {v, C, w, t};

}


int iterative_r6p(const std::vector<Eigen::Vector2d> &x, const std::vector<Eigen::Vector3d> &X, std::vector<RSCameraPose> * results, int direction){
    if (x.size() != 6 || X.size() != 6){
        return 0;
    }

    
    
    Eigen::MatrixXd xin(2,6);
    Eigen::MatrixXd Xin(3,6);
    for(size_t i = 0; i < x.size(); i++) {
        xin.col(i) = x[i];
    }

    for(size_t i = 0; i < X.size(); i++) {
        Xin.col(i) = X[i];
    }

    if(direction){
        inputSwitchDirection(Xin, xin);
    }
    
    int notFound = 1;
    int k = 0;
    double errPrev = 1e15;
    RSCameraPose result;
    
    while(notFound && k < 5){
        R6PIterResult res = R6PIter(xin, Xin, result.v); 
        
        double errNew = calcErrAlgebraicR6PFocalRadialDoubleLin(res.v, res.C, res.w, res.t,  Xin,  xin);
        if(errNew < errPrev){
            Eigen::AngleAxis eax(res.v.norm(),res.v.normalized());
            result.v = res.t;
            result.t = res.C;
            result.w = res.w;
            result.q = Eigen::Quaterniond(eax).coeffs();
            errPrev = errNew;
        }
        if(errNew < 1e-10){
            notFound = 0;
        }
        
        k++;       
    }

    if(direction){
        result = outputSwitchDirection(result);
    }

    results->push_back(result);
    return 1;
}

}