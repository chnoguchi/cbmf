
# include <vector>
# include <iostream>
# include <string>
# include <math.h>
# include <fstream>
# include <sstream>
# include <tuple>
# include <algorithm>
# include <random>
# include <chrono>
#include <unistd.h>

#include <armadillo>


using namespace std;


vector<string> split(string& input, char delimiter)
{
    istringstream stream(input);
    string field;
    vector<string> result;
    while (getline(stream, field, delimiter)) {
        result.push_back(field);
    }
    return result;
}

int get_dataset(const char* file_path, int*& ou_array, int*& ov_array, float*& y0_array)
{	

	vector<int> ou;
	vector<int> ov;
	vector<float> y0;
	ifstream ifs(file_path);
	string line;
	while(getline(ifs,line)){

		std::vector<string> strvec = split(line, ' ');

		ou.push_back( stoi(strvec.at(0)) );
		ov.push_back( stoi(strvec.at(1)) );
		y0.push_back( stof(strvec.at(2)) );


	}
	int k = ou.size();
	ou_array = new int[k];
	ov_array = new int[k];
	y0_array = new float[k];
	for (int i = 0; i < k; ++i)
	{
		ou_array[i] = ou[i];
		ov_array[i] = ov[i];
		y0_array[i] = (float)y0[i];
	}

	return k;
}

void show_vec_array(const std::vector<std::vector<int> >& array)
{
	for (int i = 0; i < array.size(); ++i)
	{
		
		// std::vector<int> array_i = array[i];
		for (int j = 0; j < array[i].size(); ++j)
		{
			printf("%i ", array[i][j]);
		}
		printf("\n");
	}
}

void show_array(int N, int M, float** array)
{
	for (int i = 0; i < N; ++i)
	{
		for (int j = 0; j < M; ++j)
		{
			printf("%f ", array[i][j]);
		}
		printf("\n");
	}
}

int maximum_value(int* array, int size)
{
	int max = array[0];
	for (int i = 0; i < size; ++i)
	{
		if (max < array[i])
		{
			max = array[i];
		}
	}
	return max;
}

float mean_matrix(float** matrix, int N, int M)
{
	float sm = 0;
	for (int i = 0; i < N; ++i)
	{
		for (int j = 0; j < M; ++j)
		{
			sm += matrix[i][j];
		}
	}
	sm = sm/(float)(N*M);
	return sm;
}

float mean_array(float* array, int size)
{
	float sm = 0;
	for (int i = 0; i < size; ++i)
	{
		sm += array[i];
	}
	return sm/(float)size;
}

float std_array(float* array, int size, float mean)
{
	float sm = 0;
	for (int i = 0; i < size; ++i)
	{
		sm += pow(array[i]-mean, 2);
	}
	sm = sm / (float)size;
	sm = sqrt(sm);
	return sm;
}


tuple<  vector< vector<int> >, vector< vector<int> >, vector< vector<int> >, vector< vector<int> >  > getNb(int* ou, int* ov,int N,int M,int k)
{
	vector< vector<int> > nbu;
	vector< vector<int> > nbv;
	vector< vector<int> > nbul;
	vector< vector<int> > nbvl;

	nbu = vector<vector<int>>(N, vector<int>(0, 0));
	nbul = vector<vector<int>>(N, vector<int>(0, 0));
	nbv = vector<vector<int>>(M, vector<int>(0, 0));
	nbvl = vector<vector<int>>(M, vector<int>(0, 0));

	for (int i = 0; i < k; ++i)
	{
		int u_ind = ou[i];
		int v_ind = ov[i];

		nbu[u_ind].push_back(v_ind);
		nbul[u_ind].push_back(i);

		nbv[v_ind].push_back(u_ind);
		nbvl[v_ind].push_back(i);
	}

	return std::forward_as_tuple(nbu, nbv, nbul, nbvl);

}

tuple<  float**, float**, int, float*  > cbmf(int N, int M, int* ou, int* ov, const vector<vector<int>>& nbu, const vector<vector<int>>& nbv, const vector<vector<int>>& nbul, const vector<vector<int>>& nbvl, int k, float* y0, int maxCnt, float gam, double conv, int R, float lam, int* te_u, int* te_v, float* te_y0, int k_te, float mean, float std)
{

	std::random_device rd;
	std::mt19937 mt(rd());
	std::uniform_real_distribution<float> rdm(0.0, 1.0);

	float *rmse_arr = new float[maxCnt];

	// allocating memory
	float **a_hat = new float*[k];
	float **b_hat = new float*[k];
	float **c_hat = new float*[k];
	float **d_hat = new float*[k];
	float **alf = new float*[k];
	float **beta = new float*[k];
	float **gamma = new float*[k];
	float **delta = new float*[k];
	for (int i = 0; i < k; ++i)
	{
		a_hat[i] = new float[R];
		b_hat[i] = new float[R];
		c_hat[i] = new float[R];
		d_hat[i] = new float[R];
		alf[i] = new float[R];
		beta[i] = new float[R];
		gamma[i] = new float[R];
		delta[i] = new float[R];
	}


	float **a_m_hat = new float*[N];
	float **b_m_hat = new float*[N];
	float **c_m_hat = new float*[M];
	float **d_m_hat = new float*[M];
	for (int i = 0; i < N; ++i)
	{
		a_m_hat[i] = new float[R];
		b_m_hat[i] = new float[R];
	}
	for (int i = 0; i < M; ++i)
	{
		c_m_hat[i] = new float[R];
		d_m_hat[i] = new float[R];
	}

	float *alf_m = new float[k];
    float *beta_m = new float[k];
    float *gamma_m = new float[k];
    float *delta_m = new float[k];

    float **u = new float*[N];
    for (int i = 0; i < N; ++i)
    {
    	u[i] = new float[R];
    }
    float **v = new float*[M];
    for (int i = 0; i < M; ++i)
    {
    	v[i] = new float[R];
    }


    auto t0 = std::chrono::system_clock::now();


    // initialize u and v
	arma::sp_mat spmx(N,M);
    for (int i = 0; i < k; ++i)
    {
    	spmx(ou[i], ov[i]) = y0[i];
    }
    arma::mat U;
	arma::vec s;
	arma::mat V;

	svds(U, s, V, spmx, R);

	s = arma::sqrt(s);
	arma::mat S = diagmat(s);

	U = U*S;
	V = V*S;


    for (int i = 0; i < N; ++i)
    {
    	for (int r = 0; r < R; ++r)
    	{
    		u[i][r] = U(i,r);
    	}
    }
    for (int i = 0; i < M; ++i)
    {
    	for (int r = 0; r < R; ++r)
    	{
    		v[i][r] = V(i,r);
    	}
    }


	for (int i = 0; i < k; ++i)
	{
		for (int j = 0; j < R; ++j)
		{
			a_hat[i][j] = rdm(mt)*5;
			b_hat[i][j] = -a_hat[i][j]*u[ou[i]][j];
			
			c_hat[i][j] = rdm(mt)*5;
			d_hat[i][j] = -c_hat[i][j]*v[ov[i]][j];
		}
	}


	for (int i = 0; i < N; ++i)
	{
		for (int r = 0; r < R; ++r)
		{
			
			float a_sm = 0;
			float b_sm = 0;
			for (const auto& nb: nbul[i])
			{
				a_sm += a_hat[nb][r];
				b_sm += b_hat[nb][r];
			}
			a_m_hat[i][r] = a_sm + lam;
			b_m_hat[i][r] = b_sm;
		}
	}
	for (int i = 0; i < M; ++i)
	{
		for (int r = 0; r < R; ++r)
		{
			
			float c_sm = 0;
			float d_sm = 0;
			for (const auto& nb: nbvl[i])
			{
				c_sm += c_hat[nb][r];
				d_sm += d_hat[nb][r];
			}
			c_m_hat[i][r] = c_sm + lam;
			d_m_hat[i][r] = d_sm;
		}
	}


	for (int i = 0; i < k; ++i)
	{
		for (int r = 0; r < R; ++r)
		{
			float vov = v[ov[i]][r];
			float vov2 = pow(vov, 2);
			float uou = u[ou[i]][r];
    		float uou2 = pow(uou, 2);
			alf[i][r] = vov2 / ( a_m_hat[ou[i]][r] - a_hat[i][r] );
			beta[i][r] = ( b_m_hat[ou[i]][r] - b_hat[i][r] ) * vov / ( a_m_hat[ou[i]][r] - a_hat[i][r] );
			gamma[i][r] = uou2 /( c_m_hat[ov[i]][r] - c_hat[i][r] );
			delta[i][r] = ( d_m_hat[ov[i]][r] - d_hat[i][r] ) * uou / ( c_m_hat[ov[i]][r] - c_hat[i][r] );
		}
	}

	float sm1;
	float sm2;
	float sm3;
	float sm4;
	for (int i = 0; i < k; ++i)
	{	
		sm1 = 0;
		sm2 = 0;
		sm3 = 0;
		sm4 = 0;
		for (int r = 0; r < R; ++r)
		{
			sm1 += alf[i][r];
			sm2 += beta[i][r];
			sm3 += gamma[i][r];
			sm4 += delta[i][r];
		}
		alf_m[i] = sm1 + 1;
		beta_m[i] = sm2;
		gamma_m[i] = sm3 + 1;
		delta_m[i] = sm4;
	}
	


	  ///////////////
     /* main loop */
    ///////////////
    int cnt;
    float rmse = 0.0;
    for (cnt = 0; cnt < maxCnt; ++cnt)
    {

    	/* u update */


    	// update alf and beta
    	float dm;
    	float vov;
    	float vov2;
    	for (int i = 0; i < k; ++i)
		{
			for (int r = 0; r < R; ++r)
			{
				vov = v[ov[i]][r];
				vov2 = pow(vov, 2);
				dm = a_m_hat[ou[i]][r] - a_hat[i][r];

				alf[i][r] = vov2 / dm;
				beta[i][r] = ( b_m_hat[ou[i]][r] - b_hat[i][r] ) * vov / dm;
			}
		}


		// update alf_m and beta_m
		float sm1;
		float sm2;
		for (int i = 0; i < k; ++i)
		{	
			sm1 = 0;
			sm2 = 0;
			for (int r = 0; r < R; ++r)
			{
				sm1 += alf[i][r];
				sm2 += beta[i][r];
			}
			alf_m[i] = sm1 + 1;
			beta_m[i] = sm2 + y0[i];
		}


    	// update a and b hat
    	for (int i = 0; i < k; ++i)
    	{	
    		for (int r = 0; r < R; ++r)
    		{
    			vov = v[ov[i]][r];
    			vov2 = pow(vov, 2);
    			dm = alf_m[i] - alf[i][r];

    			a_hat[i][r] = (1-gam)*a_hat[i][r] + gam*vov2/dm;
    			b_hat[i][r] = (1-gam)*b_hat[i][r] - gam*( beta_m[i] - beta[i][r] )*vov/dm;
    		}
    	}


    	// update a and b marginalized hat
    	float a_sm;
    	float b_sm;
    	for (int i = 0; i < N; ++i)
		{
			for (int r = 0; r < R; ++r)
			{
				a_sm = 0;
				b_sm = 0;
				for (const auto& nb: nbul[i])
				{
					a_sm += a_hat[nb][r];
					b_sm += b_hat[nb][r];
				}
				a_m_hat[i][r] = a_sm + lam;
				b_m_hat[i][r] = b_sm;
			}
		}


		// update u
		for (int i = 0; i < N; ++i)
		{
			for (int r = 0; r < R; ++r)
			{
				u[i][r] = -b_m_hat[i][r]/a_m_hat[i][r];
			}
		}



		/* update v */

		// update gamma and delta
		float uou;
		float uou2;
		for (int i = 0; i < k; ++i)
		{
			for (int r = 0; r < R; ++r)
			{
				uou = u[ou[i]][r];
	    		uou2 = pow(uou, 2);
	    		dm = c_m_hat[ov[i]][r] - c_hat[i][r];

				gamma[i][r] = uou2 /dm;
				delta[i][r] = ( d_m_hat[ov[i]][r] - d_hat[i][r] ) * uou / dm;
			}
		}


		// update gamma_m and delta_m
		for (int i = 0; i < k; ++i)
		{	
			sm1 = 0.0;
			sm2 = 0.0;
			for (int r = 0; r < R; ++r)
			{
				sm1 += gamma[i][r];
				sm2 += delta[i][r];
			}
			gamma_m[i] = sm1 + 1;
			delta_m[i] = sm2 + y0[i];
		}


		// update c and d hat
    	for (int i = 0; i < k; ++i)
    	{
    		for (int r = 0; r < R; ++r)
    		{
    			uou = u[ou[i]][r];
    			uou2 = pow(uou, 2);
    			dm = gamma_m[i] - gamma[i][r];

    			c_hat[i][r] = (1-gam)*c_hat[i][r] + gam*uou2/dm;
    			d_hat[i][r] = (1-gam)*d_hat[i][r] - gam*( delta_m[i] - delta[i][r] )*uou/dm;
    		}
    	}

    	// update c and d marginalized hat
    	float c_sm;
    	float d_sm;
    	for (int i = 0; i < M; ++i)
		{
			for (int r = 0; r < R; ++r)
			{
				
				c_sm = 0;
				d_sm = 0;
				for (const auto& nb: nbvl[i])
				{
					c_sm += c_hat[nb][r];
					d_sm += d_hat[nb][r];
				}
				c_m_hat[i][r] = c_sm + lam;
				d_m_hat[i][r] = d_sm;
			}
		}

		

		// update v
		for (int i = 0; i < M; ++i)
		{
			for (int r = 0; r < R; ++r)
			{
				v[i][r] = -d_m_hat[i][r]/c_m_hat[i][r];
			}
		}



		// calulate test rmse
		float mse = 0;
		for (int i = 0; i < k_te; ++i)
		{

			float inf_y0 = 0;
			int u_i = te_u[i];
			int v_i = te_v[i];
			for (int r = 0; r < R; ++r)
			{
				inf_y0 += u[u_i][r]*v[v_i][r];
			}
			inf_y0 = inf_y0*std + mean;
			mse += pow(inf_y0 - te_y0[i], 2);

		}
		mse = mse / (float)k_te;
		float new_rmse = sqrt(mse);
		float dif = abs(new_rmse-rmse);

		rmse_arr[cnt] = new_rmse;

		if (cnt%1 == 0)
		{
			printf("iteration:%d ", cnt);
			printf("rmse:%f\n", new_rmse);
		}

		rmse = new_rmse;
		if (dif < conv || dif != dif)
		{
			break;
		}
		


    } // end main loop

    return std::forward_as_tuple(u, v, cnt, rmse_arr);

}



tuple<  float**, float**, int, float*  > approx_cbmf(int N, int M, int* ou, int* ov, const vector<vector<int>>& nbu, const vector<vector<int>>& nbv, const vector<vector<int>>& nbul, const vector<vector<int>>& nbvl, int k, float* y0, int maxCnt, float gam, double conv, int R, float lam, int* te_u, int* te_v, float* te_y0, int k_te, float mean, float std)
{

	std::random_device rd;
	std::mt19937 mt(rd());
	std::uniform_real_distribution<float> rdm(0.0, 1.0);
	std::normal_distribution<float> g_rdm(0.0,1.0);

	float *rmse_arr = new float[maxCnt];

	// allocating memory
	float *a = new float[k];
	float *b = new float[k];
	float *c = new float[k];
	float *d = new float[k];


	float **a_hat = new float*[N];
	float **b_hat = new float*[N];
	float **u = new float*[N];
	float **c_hat = new float*[M];
	float **d_hat = new float*[M];
	float **v = new float*[M];
	for (int i = 0; i < N; ++i)
	{
		a_hat[i] = new float[R];
		b_hat[i] = new float[R];
		u[i] = new float[R];
	}
	for (int i = 0; i < M; ++i)
	{
		c_hat[i] = new float[R];
		d_hat[i] = new float[R];
		v[i] = new float[R];
	}


    arma::sp_mat spmx(N,M);
    for (int i = 0; i < k; ++i)
    {
    	spmx(ou[i], ov[i]) = y0[i];
    }
    arma::mat U;
	arma::vec s;
	arma::mat V;

	svds(U, s, V, spmx, R);

	s = arma::sqrt(s);
	arma::mat S = diagmat(s);

	U = U*S;
	V = V*S;


    for (int i = 0; i < N; ++i)
    {
    	for (int r = 0; r < R; ++r)
    	{
    		u[i][r] = U(i,r);
    	}
    }
    for (int i = 0; i < M; ++i)
    {
    	for (int r = 0; r < R; ++r)
    	{
    		v[i][r] = V(i,r);
    	}
    }


    // initialize a,b,c,d hat
	for (int i = 0; i < N; ++i)
	{
		for (int j = 0; j < R; ++j)
		{
			a_hat[i][j] = rdm(mt)+100;
			b_hat[i][j] = (a_hat[i][j])*u[i][j];
		}
	}
	for (int i = 0; i < M; ++i)
	{
		for (int j = 0; j < R; ++j)
		{
			c_hat[i][j] = rdm(mt)+100;
			d_hat[i][j] = (c_hat[i][j])*v[i][j];
		}
	}

	// initialize alf, beta, gamma, delta
	for (int i = 0; i < k; ++i)
	{
		a[i] = rdm(mt);
		b[i] = rdm(mt);
		c[i] = rdm(mt);
		d[i] = rdm(mt);
	}


	  ///////////////
     /* main loop */
    ///////////////
    printf("main loop\n");
    float vov;
	float vov2;
	float uou;
	float uou2;
    float sm1;
    float sm2;
    int cnt;
    float rmse = 0.0;
    for (cnt = 0; cnt < maxCnt; ++cnt)
    {

    	/* u update */

    	// update alf and beta
    	for (int i = 0; i < k; ++i)
		{	
			sm1 = 0;
			sm2 = 0;
			for (int r = 0; r < R; ++r)
			{
				vov = v[ov[i]][r];
				uou = u[ou[i]][r];
				vov2 = pow(vov, 2);


				// a
				sm1 += vov2/(a_hat[ou[i]][r]+lam);
				// b
				sm2 += vov*uou;
			}
			b[i] = ( y0[i]-sm2+a[i]*b[i] ) / ( 1+a[i] );
			a[i] = sm1;
			
		}



    	// update a and b hat
    	for (int i = 0; i < N; ++i)
		{
			for (int r = 0; r < R; ++r)
			{
				sm1 = 0;
				sm2 = 0;
				for (const auto& nb: nbul[i])
				{
					vov = v[ov[nb]][r];
					vov2 = pow(vov, 2);
					uou = u[i][r];

					// a_hat
					sm1 += vov2/(1+a[nb]);

					// b_hat
					sm2 += b[nb]*vov;
					sm2 += uou*vov2/(1+a[nb]);

				}
				
				b_hat[i][r] = (1-gam)*b_hat[i][r] + gam*sm2;
				a_hat[i][r] = (1-gam)*a_hat[i][r] + gam*sm1;

				// update u
				u[i][r] = b_hat[i][r]/(a_hat[i][r]+lam);

			}
		}



		/* update v */

		// update c and d
    	for (int i = 0; i < k; ++i)
		{	
			sm1 = 0;
			sm2 = 0;
			for (int r = 0; r < R; ++r)
			{
				vov = v[ov[i]][r];
				uou = u[ou[i]][r];
				uou2 = pow(uou, 2);

				// a
				sm1 += uou2/(c_hat[ov[i]][r]+lam);
				// b
				sm2 += vov*uou;
			}
			d[i] = ( y0[i]-sm2+c[i]*d[i] ) / ( 1+c[i] );
			c[i] = sm1;
			
		}


		// update c and d hat
    	for (int i = 0; i < M; ++i)
		{
			for (int r = 0; r < R; ++r)
			{
				sm1 = 0;
				sm2 = 0;
				for (const auto& nb: nbvl[i])
				{
					// vov = v[ov[nb]][r];
					vov = v[i][r];
					uou = u[ou[nb]][r];
					uou2 = pow(uou, 2);

					// a_hat
					sm1 += uou2/(1+c[nb]);

					// b_hat
					sm2 += d[nb]*uou;
					sm2 += vov*uou2/(1+c[nb]);

				}
				
				d_hat[i][r] = (1-gam)*d_hat[i][r] + gam*sm2;
				c_hat[i][r] = (1-gam)*c_hat[i][r] + gam*sm1;

				// update v
				v[i][r] = d_hat[i][r]/(c_hat[i][r]+lam);
			}
		}
		



		// calulate test rmse
		float mse = 0;
		for (int i = 0; i < k_te; ++i)
		{

			float inf_y0 = 0;
			int u_i = te_u[i];
			int v_i = te_v[i];
			for (int r = 0; r < R; ++r)
			{
				inf_y0 += u[u_i][r]*v[v_i][r];
			}
			inf_y0 = inf_y0*std + mean;

			mse += pow(inf_y0 - te_y0[i], 2);

		}
		mse = mse / (float)k_te;
		float new_rmse = sqrt(mse);
		float dif = abs(new_rmse-rmse);

		rmse_arr[cnt] = new_rmse;

		if (cnt%1 == 0)
		{
			printf("iteration:%d ", cnt);
			printf("rmse:%f\n", new_rmse);
		}

		rmse = new_rmse;
		if (dif < conv || dif != dif)
		{
			break;
		}


    } // end main loop

    return std::forward_as_tuple(u, v, cnt, rmse_arr);

}

void usage(string& doc)
{
	doc = "Usage:\n";
	doc += "   ./cbmf [-l learningrate] [-L lambda] [-R rank] [-m maxiteration] [-r trainfilename] [-t testfilename] [-o outpath] [-c convint] [-p] [-h] [-v]\n";
	doc += "\n";
	doc += "Options:\n";
	doc += "   -l   Learning rate. [default: 0.3]\n";
	doc += "   -L   Regularization parameter. [default: 3]\n";
	doc += "   -R   Rank. [default: 10]\n";
	doc += "   -m   Maximum number of iterations. [default: 100]\n";
	doc += "   -r   Filename of dataset for training. [default: dataset/ml_1m_train.txt]\n";
	doc += "   -t   Filename of dataset for test. [default: dataset/ml_1m_test.txt]\n";
	doc += "   -o   Where output files are to be placed. [default: output/]\n";
	doc += "   -c   Exponent of convergence condition. If RMSE < pow(10, convint) is satisfied, it is regarded as convergence. [default: -5]\n";
	doc += "   -p   If this option is set, ACBMF is to be performed. Without this option, CBMF is to be done.\n";
	doc += "   -h   Show help.\n";
	doc += "   -v   Show version.\n";
	doc += "\n";
	doc += "Examples:\n";
	doc += "Performing CBMF using 'dataset/ml_1m_train.txt' as training dataset and 'dataset/ml_1m_test.txt' as test dataset.\n";
	doc += "   ./cbmf -r dataset/ml_1m_train.txt -t dataset/ml_1m_test.txt\n";
	doc += "Performing ACBMF using 'dataset/ml_1m_train.txt' as training dataset and 'dataset/ml_1m_test.txt' as test dataset.\n";
	doc += "   ./cbmf -p -r dataset/ml_1m_train.txt -t dataset/ml_1m_test.txt\n";
	doc += "Showing help.\n";
	doc += "   ./cbmf -h\n";
	doc += "Showing version.\n";
	doc += "   ./cbmf -v";
}

void version(string& doc)
{
	doc = "cbmf v1.0";
}


int main(int argc, char **argv)
{
	float gam = 0.3;
	float lam = 3;
	int R = 10;
	int maxCnt = 100;
	const char* train_filename = "dataset/ml_1m_train.txt";
	const char* test_filename = "dataset/ml_1m_test.txt";
	const char* outpath = "output/";
	int is_approx = 0;
	int conv_int = -5;
	int opt;
	string doc = "";
	usage(doc);
	string vsn = "";
	version(vsn);
	while ((opt = getopt(argc, argv, "l:L:R:m:r:t:o:c:pshv")) != -1) {
        switch (opt) {
            case 'l': gam=atof(optarg); break;
            case 'L': lam=atof(optarg); break;
            case 'R': R=atoi(optarg); break;
            case 'm': maxCnt=atoi(optarg); break;
            case 'r': train_filename=optarg; break;
            case 't': test_filename=optarg; break;
            case 'o': outpath=optarg; break;
            case 'c': conv_int=atoi(optarg); break;
            case 'p': is_approx=1; break;
            case 'h':
            	printf("%s\n", doc.c_str());
            	return 0;
            case 'v':
            	printf("%s\n", vsn.c_str());
            	return 0;
            default: 
            	printf("%s\n", doc.c_str());
            	return 1;
        }
    }
    double conv = pow(10, conv_int);
	int folds = 10;

	
	printf("learning rate:%f\n", gam);
	printf("lambda:%f\n", lam);
	printf("rank:%d\n", R);
	printf("maxCnt:%d\n", maxCnt);
	printf("is_approx:%d\n", is_approx);
	printf("conv:%f\n", conv);


	int* ou_te = NULL;
	int* ov_te = NULL;
	float* y0_te = NULL;
	int* ou_tr = NULL;
	int* ov_tr = NULL;
	float* y0_tr = NULL;
	int k_te = get_dataset(test_filename,ou_te, ov_te, y0_te);
	int k_tr = get_dataset(train_filename,ou_tr, ov_tr, y0_tr);

	printf("dataset loaded\n");

	int N = maximum_value(ou_tr, k_tr)+1;
	int M = maximum_value(ov_tr, k_tr)+1;
	int K = k_tr;
	printf("N:%d\n", N);
	printf("M:%d\n", M);
	printf("K:%d\n", K);


	float mean = mean_array(y0_tr, k_tr);
	float std = std_array(y0_tr, k_tr, mean);
	printf("mean:%f\n", mean);
	printf("std:%f\n", std);
	for (int i = 0; i < k_tr; ++i)
	{
		y0_tr[i] = (y0_tr[i]-mean)/std;
	}


	vector< vector<int> > nbu;
	vector< vector<int> > nbv;
	vector< vector<int> > nbul;
	vector< vector<int> > nbvl;
	std::tie(nbu, nbv, nbul, nbvl) = getNb(ou_tr, ov_tr, N, M, K);

	printf("got nb\n");


	float** u;
	float** v;
	int cnt;
	float* rmse_arr;

	auto start = std::chrono::system_clock::now();
	if (is_approx == 0)
	{
		std::tie(u, v, cnt, rmse_arr) = cbmf(N, M, ou_tr, ov_tr, nbu, nbv, nbul, nbvl, K, y0_tr, maxCnt, gam, conv, R, lam, ou_te, ov_te, y0_te, k_te, mean, std);
	}
	else
	{
		std::tie(u, v, cnt, rmse_arr) = approx_cbmf(N, M, ou_tr, ov_tr, nbu, nbv, nbul, nbvl, K, y0_tr, maxCnt, gam, conv, R, lam, ou_te, ov_te, y0_te, k_te, mean, std);
	}
	auto end = std::chrono::system_clock::now();

	
	char gam_char[8];
	sprintf(gam_char, "%.2f", gam);
	char lam_char[8];
	sprintf(lam_char, "%.5f", lam);
	std::string filename_base = "gam=" + string(gam_char) + "_lam=" + string(lam_char) + "_R=" + to_string(R) + "_is_approx=" + to_string(is_approx);
	ofstream rmse_f(outpath + filename_base + "_rmse_approx.txt");
	for (int i = 0; i < cnt+1; ++i)
	{
		char rmse_char[16];
		sprintf(rmse_char, "%.5f", rmse_arr[i]);
		rmse_f << rmse_char;
		rmse_f << '\n';
	}
	rmse_f.close();
	

	printf("main end\n");
	
	 
	return 0;
}