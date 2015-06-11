/* 
 *a simple demo for automatically generate a maze 
  and then find and draw a shortest way for it 
  by using BFS algorithm. 
 */
#include <cstdio>
#include <cstdlib>
#include <queue>
#include <stack>
#include <ctime>
using namespace std;

const int INF = 1000000;
const int MAX_N = 200;
const int MAX_M = 200;
typedef pair<int, int> P;

int N, M;
int sx, sy;
int gx, gy;
char Maze[MAX_N][MAX_N];
int d[MAX_N][MAX_M];
P point[MAX_N][MAX_M];
char seed[3] = {'.','#','.'};
int dx[4] = {1, 0, -1, 0}, dy[4] = {0, 1, 0, -1};

int  bfs(){
    queue<P> que;
    que.push(P(sx, sy));
    d[sx][sy] = 0;

    while(que.size()){
        P p = que.front();
        que.pop();

        if(p.first == gx && p.second == gy)
            break;

        for(int i=0; i<4; i++) {
            int nx = p.first + dx[i], ny = p.second + dy[i];

            if(0<=nx && nx<N && 0<=ny && ny<M && d[nx][ny] == INF &&
                    Maze[nx][ny]!='#') {
                que.push(P(nx, ny));
                d[nx][ny] = d[p.first][p.second] + 1;
                point[nx][ny] = p;
            }
        }
    }

    return d[gx][gy];
}


void show(int res) {
   stack<P> path;
   P top = P(gx, gy);
   path.push(top);
   for(int i=0; i<res; i++)
   {
       top = path.top();
       P pre = point[top.first][top.second];
       path.push(pre);
   }
   path.pop();
   for(int i=1; i<res; i++)
   {
       top = path.top();
       path.pop();
       Maze[top.first][top.second] = '*';
   }
   for(int i=0; i<N; i++) {
       for(int j=0; j<M; j++) {
            if(Maze[i][j] == '*' || Maze[i][j] == 'G' || Maze[i][j] == 'S')
                printf("\033[32m%c\033[0m", Maze[i][j]);
            else
                if(Maze[i][j] == '#')
                    printf("\033[31m#\033[0m");
                else
                    printf("%c", Maze[i][j]);
       }
       printf("\n");

   }
}


void solve() {
    int res = bfs();
    printf("\n\n\033[32mThe shortest step is: %d\n\033[0m", res);
    if(res != INF)
        show(res);
    else
        printf("\033[31mThere is no path from S to G\033[0m\n");
}

int main(){
    srand(time(NULL));
    printf("\033[32mrow of maze--N, column of maze--M\n");
    scanf("%d %d", &N, &M);
    //char c = getchar();
    printf("Generating your Maze matrix...:\n\033[0m");
    int s = rand()%M;
    int g = rand()%M;
    Maze[0][s] = 'S';
    Maze[N-1][g] = 'G';
    for(int i=0; i<N; i++) {
        for(int j=0; j<M; j++) {
                //scanf("%c",&Maze[i][j]);
                if(Maze[i][j] != 'S' &&  Maze[i][j] != 'G')
                    Maze[i][j] = seed[rand()%sizeof(seed)];

                d[i][j] = INF;
                if(Maze[i][j] == 'S') {
                    sx = i, sy = j;
                }
                if(Maze[i][j] == 'G') {
                    gx = i, gy = j;
                }
        }
       // getchar();
    }
    printf("\033[32moriginal point S:(%d,%d)\n",sx,sy);
    printf("end  point G:(%d,%d)\033[0m\n",gx,gy);

    for(int i=0; i<N; i++){
        for(int j=0; j<M; j++) {
            if(Maze[i][j] == '#')
                printf("\033[31m#\033[0m");
            else
              if(Maze[i][j] == '.')
                printf("%c", Maze[i][j]);
              else
                printf("\033[32m%c\033[0m", Maze[i][j]);
        }
        printf("\n");
    }
    solve();
    return 0;
}
