#include <sycl/sycl.hpp>
#include <cmath>
#include <iostream>
#include <vector>
#include <cstdio>
#include <chrono>

using namespace sycl;
using std::vector;

namespace Func{
    double phi(double x){
        return x*x*x+x;
    }
    double alpha(double x){
        return 0;
    }
    double beta(double x){
        return std::exp(x)+1;
    }
    double f(double x,double t){
        return x*std::exp(t)-6*x;
    }
    double exact(double x,double t){
        return x*(x*x+std::exp(t));
    }
};
using namespace Func;


int main(){
    auto start = std::chrono::high_resolution_clock::now();
	int n,m;
	double a,h,tau,r;
	n=5000;  //时间域n等分
	m=30;    //空间域m等分
	a=1.0;
	h=1.0/m;  //空间步长
	tau=1.0/n;  //时间步长
	r=0.25;  // 与 DAC 样例保持一致，large 用例固定网比

    vector<double> x(m+1,0);
    vector<double> t(n+1,0);

    for(int i=0;i<=m;i++)
        x[i]=i*h;
    for(int i=0;i<=n;i++)
        t[i]=i*tau;

    vector<vector<double> > u(m+1,vector<double>(n+1,0));
    for(int i=0;i<=m;i++)
        u[i][0]=phi(x[i]);
    for(int i=1;i<=n;i++){
        u[0][i]=alpha(t[i]);
        u[m][i]=beta(t[i]);
    }

    vector<double> flat_u((m+1)*(n+1),0);
    for(int i=0;i<=m;i++)
        for(int j=0;j<=n;j++)
            flat_u[i*(n+1)+j]=u[i][j];
    queue q;
    double* u_device=malloc_device<double>(flat_u.size(),q);
    double* x_device=malloc_device<double>(x.size(),q);
    double* t_device=malloc_device<double>(t.size(),q);
    q.memcpy(u_device,flat_u.data(),sizeof(double)*flat_u.size()).wait();
    q.memcpy(x_device,x.data(),sizeof(double)*x.size()).wait();
    q.memcpy(t_device,t.data(),sizeof(double)*t.size()).wait();
    for(int k=0;k<n;k++){
        int step=m-1;
        if(step>0)
        q.parallel_for(range<1>(step),[=](id<1>idx){
            int i=1+idx[0];
            u_device[i*(n+1)+k+1]=r*u_device[(i-1)*(n+1)+k]+(1-2*r)*u_device[i*(n+1)+k]+r*u_device[(i+1)*(n+1)+k];
        }).wait();
    }

    q.memcpy(flat_u.data(),u_device,sizeof(double)*flat_u.size()).wait();
    for(int i=0;i<=m;i++)for(int j=0;j<=n;j++)u[i][j]=flat_u[i*(n+1)+j];
	 int number=int(0.4/h);
	//  for(int k=int(0.2/tau);k<=n;k=k+int(0.2/tau)){
	//  	printf("(x,t)=(%.1f,%.1f), y=%.2f, exact=%.3f, err=%.3e.\n",x[number],t[k],u[number][k],exact(x[number],t[k]),std::fabs(u[number][k]-exact(x[number],t[k])));
	//  }

    //  for(int k=int(0.2/tau);k<=n;k=k+int(0.2/tau)){
    //     printf("(x,t)=(%.1f,%.1f), y=%.2f, exact=%.3f.\n",x[number],t[k],u[number][k],exact(x[number],t[k]));
    // }
    std::cout << "{";
    for(int i = 1; i < 2; i++){
        
        for(int j = 0; j < n+1; j++){
            std::cout  << u[i][j] ;
            if (j < n) std::cout << ", ";
        }
        
        
    }
    std::cout << "}" << std::endl;

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration_sycl = end - start;
    //std::cout << "SYCL code time: " << duration_sycl.count() << " seconds" << std::endl;
    free(u_device,q);free(x_device,q);free(t_device,q);
    return 0;
}
