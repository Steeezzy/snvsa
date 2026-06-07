package helix;

import java.net.*;
import java.io.*;

public class EnvironmentBridge {
    private final String host;
    private final int port;
    private Socket socket;
    private PrintWriter out;
    private BufferedReader in;
    private double successRate = 0.0;
    private int episodes = 0;

    public EnvironmentBridge(String host, int port) {
        this.host = host;
        this.port = port;
    }

    public void connect() throws Exception {
        socket = new Socket(host, port);
        out = new PrintWriter(socket.getOutputStream(), true);
        in  = new BufferedReader(
            new InputStreamReader(socket.getInputStream()));
    }

    // returns [obs, reward]
    public double[] step(int action) throws Exception {
        out.println(action);
        String response = in.readLine();
        if (response == null) return new double[]{0.2, 0.0};
        var json = new org.json.JSONObject(response);
        successRate = json.optDouble("success_rate", 0.0);
        episodes    = json.optInt("episodes", 0);
        return new double[]{
            json.getDouble("obs"),
            json.getDouble("reward")
        };
    }

    public double getSuccessRate() { return successRate; }
    public int getEpisodes() { return episodes; }

    public void close() throws Exception { socket.close(); }
}
