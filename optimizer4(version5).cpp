#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstdio>
#include <mpi.h>

using namespace std;

typedef long long ll;
const int cache_sz = 32 * 1024, N = 1000001;
bool vis[max(cache_sz, N)];

ll pos[N], pri[N];
inline ll mmax(ll a, ll b) { return a < b ? b : a; }
inline ll mmin(ll a, ll b) { return a < b ? a : b; }
int main(int argc, char** argv) {
	MPI_Init(&argc, &argv);
	int id, cnt;
	MPI_Comm_rank(MPI_COMM_WORLD, &id);
	MPI_Comm_size(MPI_COMM_WORLD, &cnt);
	MPI_Barrier(MPI_COMM_WORLD);
	double elapsed_time = -MPI_Wtime();
	MPI_Barrier(MPI_COMM_WORLD);

	ll n = atoll(argv[1]);
	ll low_val, high_val;
	ll q = (n + 1) / cnt, r = (n + 1) % cnt;
	if (id < r) {
		low_val = id * (q + 1);
		high_val = low_val + (q + 1);
	}
	else {
		low_val = r * (q + 1) + (id - r) * q;
		high_val = low_val + q;
	}

	if (id == 0) low_val = 2;

	const int sqrtrb = sqrt(high_val);
	int tot = 0;

	for (int i = 3; i <= sqrtrb; i += 2) {
		if (!vis[i]) {
			pri[++tot] = i;
			pos[tot] = (i * mmax(((low_val - 1) / i + 1) | 1, 1ll * i)) >> 1;
		}
		for (int j = 1; j <= tot && i*pri[j] <= sqrtrb; ++j) {
			vis[i*pri[j]] = 1;
			if (i%pri[j] == 0)break;
		}
	}



	ll count = (high_val >> 1) - (low_val >> 1);
	if (!id) count++;
	for (ll l = low_val, r; l < high_val; l = r) {
		r = mmin(high_val, l + (cache_sz << 1));
		const ll bl = l >> 1, br = r >> 1;
		memset(vis, 0, br - bl);
		for (int i = 1; i <= tot; ++i) {
			const int p = pri[i];
			ll j = pos[i];
			for (; j < br; j += p)
				vis[j - bl] = 1;
			pos[i] = j;
		}
		for (ll j = bl; j < br; ++j)
			count -= vis[j - bl];
	}

	ll global_count = 0;
	if (cnt > 1)
		MPI_Reduce(&count, &global_count, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
	else
		global_count = count;
	elapsed_time += MPI_Wtime();

	if (id == 0) {
		printf("There are %lld primes less than or equal to %lld\n",
			global_count, n);
		printf("SIEVE (%d) %10.6f\n", cnt, elapsed_time);
	}
	MPI_Finalize();
	return 0;
}