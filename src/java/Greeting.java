/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

import java.io.IOException;
import java.io.PrintWriter;
import java.util.logging.Level;
import java.util.logging.Logger;
import javax.servlet.ServletException;
import javax.servlet.annotation.WebServlet;
import javax.servlet.http.HttpServlet;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;
import org.json.JSONException;
import org.json.JSONObject;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.core.Debug;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Normalize;

/**
 *
 * @author lagg99
 */
@WebServlet(urlPatterns = {"/Greeting"})
public class Greeting extends HttpServlet {

    /**
     * Processes requests for both HTTP <code>GET</code> and <code>POST</code>
     * methods.
     *
     * @param request servlet request
     * @param response servlet response
     * @throws ServletException if a servlet-specific error occurs
     * @throws IOException if an I/O error occurs
     */
    protected void processRequest(HttpServletRequest request, HttpServletResponse response)
            throws ServletException, IOException {
        response.setContentType("text/html;charset=UTF-8");
        try (PrintWriter out = response.getWriter()) {

            double acc_x = Double.parseDouble(request.getParameter("acc_x"));
            double acc_y = Double.parseDouble(request.getParameter("acc_y"));
            double acc_z = Double.parseDouble(request.getParameter("acc_z"));
            double gyro_x = Double.parseDouble(request.getParameter("gyro_x"));
            double gyro_y = Double.parseDouble(request.getParameter("gyro_y"));
            double gyro_z = Double.parseDouble(request.getParameter("gyro_z"));
            double magn_x = Double.parseDouble(request.getParameter("magn_x"));
            double magn_y = Double.parseDouble(request.getParameter("magn_y"));
            double magn_z = Double.parseDouble(request.getParameter("magn_z"));
            int steps = Integer.parseInt(request.getParameter("steps"));

            // acc_x=0.358835,acc_y=0.864053,acc_z=-0.005573,gyro_x=-0.1986,gyro_y=-0.122358,gyro_z=-0.045218,magn_x=-0.1986,magn_y=-0.122358,magn_z=-0.045218
            JSONObject jsonObject = new JSONObject();
            try {
                String result = main(acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z, magn_x, magn_y, magn_z, steps);
                jsonObject.put("Greeting", result);
                out.println(jsonObject);
            } catch (JSONException ex) {
                Logger.getLogger(Greeting.class.getName()).log(Level.SEVERE, null, ex);
            }

        }
    }

    public String main(double acc_x, double acc_y, double acc_z,
            double gyro_x, double gyro_y, double gyro_z,
            double magn_x, double magn_y, double magn_z, int steps) {
        String result = "";
        try {
            DataSource source = new DataSource("E:/Master/DataSet/TEST/datamodel3.arff");
            Instances dataset = source.getDataSet();
            dataset.setClassIndex(dataset.numAttributes() - 1);
            int numClasses = dataset.numClasses();
            System.out.println(numClasses);
            for (int i = 0; i < numClasses; i++) {
                String classValue = dataset.classAttribute().value(i);
                System.out.println("the " + i + "th class value : " + classValue);
            }

            dataset.randomize(new Debug.Random(1));// if you comment this line the accuracy of the model will be droped from 96.6% to 80%

            //Normalize dataset
            Normalize normalize = new Normalize();
            normalize.setInputFormat(dataset);
            Instances newdata = Filter.useFilter(dataset, normalize);

            // load test data
            DataSource source2 = new DataSource("E:/Master/DataSet/TEST/datamodel3.arff");
            Instances testdata = source2.getDataSet();
            testdata.setClassIndex(testdata.numAttributes() - 1);

        // -0.086055,-0.361837,-0.64796,0.057196,0.075941,0.129704,-282.665838,264.22574,147.078061,walking --> sitting
        // 0.004441,-0.006493,0.998635,-0.00011,0.000041,0.000714,-241.053011,195.015964,26.104802,sitting --> sitting
        //0.358835,0.864053,-0.005573,-0.1986,-0.122358,-0.045218,-0.1986,-0.122358,-0.045218,running --> standding
        
        // add a new instance to which we apply a prediction
        //addNewInstance(testdataset, 0.358835,0.864053,-0.005573,-0.1986,-0.122358,-0.045218,-0.1986,-0.122358,-0.045218);
        addNewInstance(testdata, acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z, magn_x, magn_y, magn_z, steps);
        System.out.println("last instn testdataset : \r\n " + testdata.lastInstance());

// make prediction of new instance by neural networks classifier
//        Instances labeled = new Instances(testdataset);
//        double clsLabel = ann.classifyInstance(testdataset.lastInstance());
//        labeled.lastInstance().setClassValue(clsLabel);
//        System.out.println(clsLabel);
//        System.out.println("from the classifier hereeee: " + labeled.lastInstance().stringValue(9));
//        System.out.println("last instn labeled : \r\n " + labeled.lastInstance());
            
            // make prediction of a new instance using saved model
            // load the model
            Classifier cls = (Classifier) weka.core.SerializationHelper.read("E:/Master/DataSet/TEST/modelMer.model");

            Instances labeled2 = new Instances(testdata);
            double value = cls.classifyInstance(testdata.lastInstance());
            labeled2.lastInstance().setClassValue(value);
            System.out.println(value);
            System.out.println("from the model hereeee: " + labeled2.lastInstance().stringValue(10));
            System.out.println("last instn labeled 2 : \r\n " + labeled2.lastInstance());
            result = labeled2.lastInstance().stringValue(10);
            
            // make prediction of the testdataset useing the saved model
//            for (int j = 0; j < testdata.numInstances(); j++) {
//                double actualClass = testdata.instance(j).classValue();
//                String actual = testdata.classAttribute().value((int) actualClass);
//                Instance newInst = testdata.instance(j);
//                //			System.out.println("actual class:"+newInst.stringValue(newInst.numAttributes()-1));
//                double preNN = cls.classifyInstance(newInst);
//                String predString = testdata.classAttribute().value((int) preNN);
//                System.out.println("actuel : " + actual + " ,predication   " + predString);
//            }
        } catch (Exception ex) {
            Logger.getLogger(Greeting.class.getName()).log(Level.SEVERE, null, ex);
        }
        return result;
    }

    public static void addNewInstance(Instances testdataset, double acc_x, double acc_y, double acc_z,
            double gyro_x, double gyro_y, double gyro_z,
            double magn_x, double magn_y, double magn_z, int steps) {

        Instance inst = new Instance(11);
        inst.setValue(testdataset.attribute(0), acc_x);
        inst.setValue(testdataset.attribute(1), acc_y);
        inst.setValue(testdataset.attribute(2), acc_z);
        inst.setValue(testdataset.attribute(3), gyro_x);
        inst.setValue(testdataset.attribute(4), gyro_y);
        inst.setValue(testdataset.attribute(5), gyro_z);
        inst.setValue(testdataset.attribute(6), magn_x);
        inst.setValue(testdataset.attribute(7), magn_y);
        inst.setValue(testdataset.attribute(8), magn_z);
        inst.setValue(testdataset.attribute(9), steps);

        // add
        testdataset.add(inst);
    }

    // <editor-fold defaultstate="collapsed" desc="HttpServlet methods. Click on the + sign on the left to edit the code.">
    /**
     * Handles the HTTP <code>GET</code> method.
     *
     * @param request servlet request
     * @param response servlet response
     * @throws ServletException if a servlet-specific error occurs
     * @throws IOException if an I/O error occurs
     */
    @Override
    protected void doGet(HttpServletRequest request, HttpServletResponse response)
            throws ServletException, IOException {
        processRequest(request, response);
    }

    /**
     * Handles the HTTP <code>POST</code> method.
     *
     * @param request servlet request
     * @param response servlet response
     * @throws ServletException if a servlet-specific error occurs
     * @throws IOException if an I/O error occurs
     */
    @Override
    protected void doPost(HttpServletRequest request, HttpServletResponse response)
            throws ServletException, IOException {
        processRequest(request, response);
    }

    /**
     * Returns a short description of the servlet.
     *
     * @return a String containing servlet description
     */
    @Override
    public String getServletInfo() {
        return "Short description";
    }// </editor-fold>

}
