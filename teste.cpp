projeto/
│── main.cpp
│── haarcascade_frontalface_default.xml
│── modelo.yml
│── dataset/
#include <opencv2/opencv.hpp>
#include <opencv2/face.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <ctime>
#include <filesystem>

using namespace cv;
using namespace cv::face;
using namespace std;
namespace fs = std::filesystem;

struct Pessoa {
    int id;
    string nome;
    bool presente;
};

vector<Pessoa> lista;
int proximoId = 1;

// --- Função para registrar log em arquivo ---
void registrarLog(const string& acao) {
    ofstream arquivo("logs_presenca.txt", ios::app);
    if (arquivo.is_open()) {
        time_t agora = time(0);
        char* dt = ctime(&agora);
        arquivo << "[" << dt << "] " << acao << "\n";
        arquivo.close();
    }
}

// --- Captura imagens do rosto para treinamento ---
void capturarImagens(int id) {
    CascadeClassifier face_cascade;
    if (!face_cascade.load("haarcascade_frontalface_default.xml")) {
        cerr << "Erro ao carregar Haar Cascade!\n";
        return;
    }

    VideoCapture cap(0);
    if (!cap.isOpened()) {
        cerr << "Erro ao abrir camera!\n";
        return;
    }

    int contador = 0;
    fs::create_directories("dataset");

    while (contador < 20) { // captura 20 imagens por pessoa
        Mat frame, gray;
        cap >> frame;
        cvtColor(frame, gray, COLOR_BGR2GRAY);

        vector<Rect> faces;
        face_cascade.detectMultiScale(gray, faces);

        for (auto &face : faces) {
            Mat faceROI = gray(face);
            string filename = "dataset/user_" + to_string(id) + "_" + to_string(contador) + ".jpg";
            imwrite(filename, faceROI);
            rectangle(frame, face, Scalar(255,0,0), 2);
            contador++;
            cout << "Imagem capturada: " << filename << endl;
        }

        imshow("Captura de rosto", frame);
        if (waitKey(100) == 'q') break;
    }

    cap.release();
    destroyAllWindows();
}

// --- Treinamento do modelo LBPH ---
void treinarModelo() {
    vector<Mat> faces;
    vector<int> labels;

    for (auto& entry : fs::directory_iterator("dataset")) {
        string path = entry.path().string();
        Mat img = imread(path, IMREAD_GRAYSCALE);

        if (!img.empty()) {
            // Extrair ID do nome do arquivo: user_ID_num.jpg
            int id = stoi(path.substr(path.find("user_") + 5, path.find("", path.find("user") + 5) - (path.find("user_") + 5)));
            faces.push_back(img);
            labels.push_back(id);
        }
    }

    if (faces.empty()) {
        cerr << "Nenhuma imagem encontrada para treinamento!\n";
        return;
    }

    Ptr<LBPHFaceRecognizer> model = LBPHFaceRecognizer::create();
    model->train(faces, labels);
    model->save("modelo.yml");
    cout << "Treinamento concluído e salvo em modelo.yml\n";
}

// --- Reconhecimento facial ---
void reconhecimentoFacial() {
    CascadeClassifier face_cascade;
    if (!face_cascade.load("haarcascade_frontalface_default.xml")) {
        cerr << "Erro ao carregar Haar Cascade!\n";
        return;
    }

    Ptr<LBPHFaceRecognizer> model = LBPHFaceRecognizer::create();
    try {
        model->read("modelo.yml");
    } catch (...) {
        cerr << "Erro: treine o modelo primeiro!\n";
        return;
    }

    VideoCapture cap(0);
    if (!cap.isOpened()) {
        cerr << "Erro ao abrir camera!\n";
        return;
    }

    cout << "Pressione 'q' para sair...\n";

    while (true) {
        Mat frame, gray;
        cap >> frame;
        cvtColor(frame, gray, COLOR_BGR2GRAY);

        vector<Rect> faces;
        face_cascade.detectMultiScale(gray, faces);

        for (auto &face : faces) {
            Mat faceROI = gray(face);

            int idPredito = -1;
            double confianca = 0.0;
            model->predict(faceROI, idPredito, confianca);

            if (idPredito != -1 && confianca < 80) { // quanto menor a confiança, mais seguro
                for (auto &p : lista) {
                    if (p.id == idPredito) {
                        if (!p.presente) {
                            p.presente = true;
                            registrarLog("Check-in facial: " + p.nome + " (ID " + to_string(p.id) + ")");
                        }
                        putText(frame, p.nome + " (ID " + to_string(p.id) + ")", Point(face.x, face.y - 10),
                                FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0,255,0), 2);
                        break;
                    }
                }
            } else {
                putText(frame, "Desconhecido", Point(face.x, face.y - 10),
                        FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0,0,255), 2);
            }

            rectangle(frame, face, Scalar(255,0,0), 2);
        }

        imshow("Reconhecimento Facial", frame);
        if (waitKey(30) == 'q') break;
    }

    cap.release();
    destroyAllWindows();
}

int main() {
    int opcao;
    do {
        cout << "\n=== Sistema de Check-in com OpenCV ===\n";
        cout << "1. Registrar pessoa (capturar rosto)\n";
        cout << "2. Treinar modelo facial\n";
        cout << "3. Fazer check-in facial\n";
        cout << "4. Ver lista de presenca\n";
        cout << "5. Sair\n";
        cout << "Escolha: ";
        cin >> opcao;
        cin.ignore();

        if (opcao == 1) {
            Pessoa p;
            p.id = proximoId++;
            cout << "Digite o nome: ";
            getline(cin, p.nome);
            p.presente = false;
            lista.push_back(p);
            registrarLog("Pessoa registrada: " + p.nome + " (ID " + to_string(p.id) + ")");
            capturarImagens(p.id);
        }
        else if (opcao == 2) {
            treinarModelo();
        }
        else if (opcao == 3) {
            reconhecimentoFacial();
        }
        else if (opcao == 4) {
            cout << "\n--- Lista de Presenca ---\n";
            for (auto &p : lista) {
                cout << "ID: " << p.id << " | " << p.nome 
                     << " - " << (p.presente ? "Presente" : "Ausente") << endl;
            }
        }

    } while (opcao != 5);

    return 0;
}